import argparse
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import ray
import wandb
import yaml
import traceback
from copy import deepcopy
import pickle
import os
import hashlib
import json

from utils import ProgressBar, override_config, experiments
from simple_mamba import MambaLM, MambaLMConfig
from datasets import InductionHead, DynamicCategoricalDataset

if not torch.cuda.is_available():
    raise NotImplementedError("Cannot run on CPU!")

device = torch.device('cuda')


def train(config, model, data_loader, optimizer, mask, save_one_step_file=None):
    losses = []
    for epoch in range(config["train"]["num_epochs"]):
        avg_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        total_correct_sequences = 0
        first_token_correct_count = 0
        last_token_correct_count = 0
        for data, labels in data_loader:
            data = data.to(device).long()  # Ensure data is on the correct device and dtype
            labels = labels.to(device).long()  # Ensure labels are on the correct device and converted to long

            # Forward pass
            logits = model(data)  # [batch_size, seq_len, cat_num]

            # Compute loss
            loss = F.cross_entropy(
                logits[:, mask:, :].reshape(-1, config["dataset"]["n_categories"]),
                labels[:, mask:].reshape(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            # Calculate predictions
            _, predicted = torch.max(logits, dim=2)  # [batch_size, seq_len]

            # Mask to focus only on relevant positions
            relevant_labels = labels[:, mask:]
            relevant_predicted = predicted[:, mask:]

            # Calculate correct predictions per token
            correct_tokens = (relevant_predicted == relevant_labels).sum()
            total_correct_tokens += correct_tokens.item()
            total_tokens += relevant_labels.numel()  # Total number of evaluated tokens

            # Calculate correct predictions per sequence
            correct_sequences = (relevant_predicted == relevant_labels).all(dim=1).sum()
            total_correct_sequences += correct_sequences.item()

            # Accuracy for the first and last tokens in the sequence
            first_token_correct_count += (relevant_predicted[:, 0] == relevant_labels[:, 0]).sum().item()
            last_token_correct_count += (relevant_predicted[:, -1] == relevant_labels[:, -1]).sum().item()

            if save_one_step_file is not None:
                os.makedirs(os.path.dirname(save_one_step_file), exist_ok=True)
                torch.save(model.state_dict(), f'{save_one_step_file}.pth')
                with open(f'{save_one_step_file}.pkl', 'wb') as file:
                    pickle.dump([logits, loss], file)
                return

        total_sequences = sum(len(labels) for _, labels in data_loader)
        avg_loss /= len(data_loader)
        avg_accuracy_per_token = total_correct_tokens / total_tokens
        avg_accuracy_per_sequence = total_correct_sequences / total_sequences
        first_token_accuracy = first_token_correct_count / total_sequences
        last_token_accuracy = last_token_correct_count / total_sequences

        losses.append(avg_loss)
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "min_loss": np.min(losses),
            "avg_accuracy_per_token": avg_accuracy_per_token,
            "avg_accuracy_per_sequence": avg_accuracy_per_sequence,
            "first_token_accuracy": first_token_accuracy,
            "last_token_accuracy": last_token_accuracy
        })

        if avg_loss < config["train"]["stop_on_loss"]:
            break


def get_dataset_mask(data_config):
    if data_config["name"] == "induction_head":
        dataset = InductionHead(data_config["epoch_size"],
                                data_config["seq_len"],
                                data_config["n_categories"],
                                data_config["num_triggers"],
                                data_config["induction_len"],
                                data_config["auto_regressive"])
        mask = -data_config["induction_len"]

    elif data_config["name"] == "delay":
        dataset = DynamicCategoricalDataset(data_config["epoch_size"],
                                            data_config["extra"] + data_config["lag"],
                                            data_config["n_categories"],
                                            data_config["lag"],
                                            data_config["auto_regressive"],
                                            data_config["copy_token"])
        mask = data_config["lag"]

    else:
        raise NotImplementedError

    return dataset, mask


@ray.remote(num_gpus=0.5)
def run_experiment(config, progress_bar_actor, file_path):
    try:
        wandb_config = config["wandb"]
        model_config = config["model"]
        data_config = config["dataset"]
        train_config = config["train"]

        wandb.init(
            entity=wandb_config["entity"],
            project=wandb_config["project"],
            config=config,
            name=f"{model_config['ssm_type']}"
        )

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        mamba_config = MambaLMConfig(**model_config)
        model = MambaLM(mamba_config).to(device)

        dataset, mask = get_dataset_mask(data_config)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=train_config["batch_size"],
                                                  shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=train_config["lr"])

        if file_path is not None:
            json_data = json.dumps(config, sort_keys=True)  # sort_keys ensures consistent order
            # Generate the hash (using SHA256 as an example)
            hash_object = hashlib.sha256(json_data.encode('utf-8'))
            file_path = f"{file_path}/{hash_object.hexdigest()}"

        train(config, model, data_loader, optimizer, mask, save_one_step_file=file_path)

        for i in range(6, 21):
            test_data_config = deepcopy(data_config)
            test_data_config["seq_len"] = 2 ** i
            test_dataset, mask = get_dataset_mask(test_data_config)
            test_data_loader = torch.utils.data.DataLoader(test_dataset)
            # test_ext(model, test_data_loader, mask, test_data_config["seq_len"])

    except Exception as e:
        print(progress_bar_actor, "fail:", traceback.format_exc())
    progress_bar_actor.update.remote()
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="experiment config file")
    parser.add_argument('--overrides', nargs='*', default=[],
                        help='Provide overrides as key=value pairs (e.g., model.ssm_type="S4D-Complex").')
    parser.add_argument("--file", type=str, default=None, help="One step file to save")
    config = parser.parse_args().config
    overrides = parser.parse_args().overrides
    print(f"\nUsing config {config}")
    print(f"\nOverrides: {overrides}")

    with open("configs/" + config) as stream:
        try:
            base_config = yaml.safe_load(stream)
            base_config = override_config(base_config, overrides)
        except yaml.YAMLError as exc:
            raise RuntimeError(exc)

    ray.init(num_cpus=64, ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor
    if "wandb" in base_config and "api_key" in base_config["wandb"]:
        wandb.login(key=base_config["wandb"]["api_key"])

    # You can modify the values here to run in parallel using ray
    tasks = []
    settings_options = [
        ["d_state", [16]],
        ["seed", [4]],
        # ["dataset.induction_len", [16, 32, 64, 128, 255]],
        # ["dataset.auto_regressive", [True]],
        # ["model.S4_init", ["diag-lin", "legs", "diag-real", "diag-legs", "diag-random"]],
        ["model.bias", [False]],
        ["model.B_is_selective", [True, False]],
        ["model.C_is_selective", [True, False]],
        ["model.dt_is_selective", [False, True]],
        ["model.channel_sharing", [False]],
        ["model.ssm_type", ["S6-Real", "S6-Complex"]],
    ]
    for config in experiments(settings_options):
        config.update({"comment": ""})
        config = override_config(base_config, [f"{k}={v}" for k, v in config.items()])
        print("\nCONFIG:")
        print(yaml.dump(config))
        tasks.append(run_experiment.remote(config, progress_bar_actor, file_path=parser.parse_args().file))
    pb.set_total(len(tasks))
    pb.print_until_done()

    print("finished running all")


if __name__ == "__main__":
    main()
