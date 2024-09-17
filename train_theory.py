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

    opt = optimizers[opt]
    lr = float(lr)
    r_min = float(r_min)
    num_epochs = int(num_epochs)
    seed = int(seed)
    N = int(N)
    if L == "N":
        L = N

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",
                        type=str,
                        required=True,
                        help="Name of the experiment to run. Options: <complex, real>")
    parser.add_argument('--project_name',
                        type=str,
                        required=False,
                        default="TheoreticalSSM",
                        help='The name of the wandb project to save results in')
    parser.add_argument('--num_cpus', type=int, default=4, help='Number of CPUs to use')

    experiment = parser.parse_args().experiment

    if experiment == "complex":
        settings_options = [
            ["seed", [0, 1, 2]],
            ["task_name", ["random", "delay", "oscillation"]],
            ["N", [256, 32, 64, 128]],
            ["L", ["N"]],
        ]
    elif experiment == "real":
        settings_options = [
            ["seed", [0, 1, 2]],
            ["task_name", ["random", "delay", "oscillation"]],
            ["r_min", [0.0, 0.99]],
            ["lr", [0.0001, 0.00001, 0.000001]],
            ["opt", ["adam", "adamw", "radam"]]
        ]
    else:
        raise Exception(f"Unknown experiment name: '{experiment}'. Expected 'complex' or 'real'.")
    
    settings_options.append(['wandb_project_name', [parser.parse_args().project_name]])

    config_path = f"./configs/theory/theory_{experiment}.yaml"

    with open(config_path) as stream:
        base_config = yaml.safe_load(stream)

    ray.init(num_cpus=parser.parse_args().num_cpus, ignore_reinit_error=True)

    tasks = []
    for config in experiments(settings_options):
        config = override_config(base_config, [f"{k}={v}" for k, v in config.items()])
        print("\nCONFIG:")
        print(yaml.dump(config))
        tasks.append(train_multiprocess.remote(**config))

    ray.get(tasks)
    print("finished running all")


if __name__ == "__main__":
    main()
