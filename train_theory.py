import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import ray
import argparse
import yaml

from theory_ssm.theory_ssm import VanillaSSM, get_target_imp_res, impulse_response
from utils import override_config, experiments


optimizers = {"adam": optim.Adam,
              "adamw": optim.AdamW,
              "radam": optim.RAdam}

def train(N, complex_option, seed, task_name, L, lr,
          r_min, num_epochs,
          opt, BC_std=0.001, use_wandb=False, wandb_project_name=""
          ):
    torch.manual_seed(seed)

    if L == "N":
        L = N
    opt = optimizers[opt]

    ssm = VanillaSSM(N, complex_option=complex_option, BC_std=BC_std, r_min=r_min)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = opt(ssm.parameters(), lr=lr)
    opt_name = optimizer.__class__.__name__

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    if use_wandb:
        # Create a config dictionary with the parameters
        config = {
            "N": N,
            "complex_option": complex_option,
            "L": L,
            "BC_std": BC_std,
            "num_epochs": num_epochs,
            "opt": opt.__name__,
            "seed": seed,
            "opt": opt_name,
            "task_name": task_name,
            "lr": lr,
            "r_min": r_min
        }
        run_name = f"task_{task_name}_L_{L}_complex_option_{complex_option}_N_{N}"
        wandb.init(
            project=wandb_project_name,
            name=run_name,
            config=config,
        )

    target_imp_res = get_target_imp_res(task_name, L)
    target_imp_res /= torch.linalg.norm(target_imp_res)  # normalizing to be 1

    # Initialize lists to store the loss values
    min_loss = 1e100

    for epoch in range(num_epochs):
        ssm.train()
        imp_res = impulse_response(ssm, L)
        loss = criterion(imp_res, target_imp_res)
        loss.backward()
        optimizer.step()
        if min_loss > loss.detach().numpy():
            min_loss = loss.detach().numpy()
        scheduler.step()
        if use_wandb and (epoch + 1) % 500 == 0:
            wandb.log(
                data={"epoch": epoch + 1,
                      "loss": loss,
                      "min_loss": min_loss,
                      },
                step=epoch
            )
    wandb.finish(quiet=True)
    print(f"Finished task: seed={seed}, L={L}, opt={opt.__name__}, lr={lr}, task={task_name}")


@ray.remote(num_cpus=1)
def train_multiprocess(**kwargs):
    train(**kwargs)


# #%%
# BC_std = 0.001  # Standard deviation for B and C initialization
# num_epochs = 500000
# seeds = [0, 1, 2, ]
# task_names = ["random", "delay", "oscillation"]
# lrs = [0.0001, 0.00001, 0.000001]
# r_mins = [0, 0.99]
#
#
# tasks = []
#
# # Experiments for complex parametrization
# complex_option = True
# for seed in seeds:
#     for L in [256, 32, 64, 128]:
#         for task_name in task_names:
#             for cur_opt in [optim.Adam, ]:
#                 torch.manual_seed(seed)
#
#                 N = L
#                 task = train_multiprocess.remote(N=N,
#                                                  complex_option=complex_option,
#                                                  seed=seed, L=L,
#                                                  task_name=task_name,
#                                                  opt=cur_opt,
#                                                  num_epochs=num_epochs,
#                                                  use_wandb=True,
#                                                  lr=0.00001,
#                                                  r_min=0.99
#                                                  )
#                 tasks.append(task)
#
# # Experiments for real parametrization
# complex_option = False
# for r_min in r_mins:
#     for seed in seeds:
#         for L in [32, ]:
#             for cur_opt in [optim.Adam, optim.AdamW, optim.RAdam]:
#                 for lr in lrs:
#                     for task_name in task_names:
#                         task = torch.manual_seed(seed)
#                         N = 1024
#                         task = train_multiprocess.remote(N=N,
#                                                          complex_option=complex_option,
#                                                          seed=seed, L=L,
#                                                          task_name=task_name,
#                                                          opt=cur_opt,
#                                                          num_epochs=num_epochs,
#                                                          use_wandb=True,
#                                                          lr=lr,
#                                                          r_min=r_min
#                                                          )
#                         tasks.append(task)
#
# results = ray.get(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="experiment config file")
    parser.add_argument('--overrides', nargs='*', default=[],
                        help='Provide overrides as key=value pairs (e.g., lr=0.01).')
    parser.add_argument('--num_cpus', type=int, default=4, help='Number of CPUs to use')
    config = parser.parse_args().config
    overrides = parser.parse_args().overrides
    print(f"\nUsing config {config}")
    print(f"\nOverrides: {overrides}")

    with open("configs/theory/" + config) as stream:
        try:
            base_config = yaml.safe_load(stream)
            base_config = override_config(base_config, overrides)
        except yaml.YAMLError as exc:
            raise RuntimeError(exc)

    ray.init(num_cpus=parser.parse_args().num_cpus, ignore_reinit_error=True)
    if "wandb" in base_config and "api_key" in base_config["wandb"]:
        wandb.login(key=base_config["wandb"]["api_key"])

    # You can modify the values here to run in parallel using ray
    tasks = []

    # settings for complex
    settings_options = [
        ["seed", [0, 1, 2]],
        ["task_name", ["random", "delay", "oscillation"]],
        ["N", [256, 32, 64, 128]],
        ["L", ["N"]],
    ]

    # settings for real
    # settings_options = [
    #     ["seed", [0, 1, 2]],
    #     ["task_name", ["random", "delay", "oscillation"]],
    #     ["r_min", [0, 0.99]],
    #     ["lr", [0.0001, 0.00001, 0.000001]],
    #     ["opt", ["adam", "adamw", "radam"]]
    # ]

    for config in experiments(settings_options):
        config = override_config(base_config, [f"{k}={v}" for k, v in config.items()])
        print("\nCONFIG:")
        print(yaml.dump(config))
        tasks.append(train_multiprocess.remote(**config))

    ray.get(tasks)
    print("finished running all")


if __name__ == "__main__":
    main()
