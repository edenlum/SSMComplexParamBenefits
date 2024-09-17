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


def train(config, model, data_loader, optimizer, mask):
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


@ray.remote(num_gpus=1)
def run_experiment(config, progress_bar_actor):
    try:
        wandb_config = config["wandb"]
        model_config = config["model"]
        data_config = config["dataset"]
        train_config = config["train"]

        wandb.init(
            entity=wandb_config.get("entity", None),
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

        train(config, model, data_loader, optimizer, mask)

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
    parser.add_argument("--ablation", action='store_true', default=False, required=False, help="Run ablation study")
    parser.add_argument('--num_cpus', type=int, default=4, help='Number of CPUs to use')
    parser.add_argument('--project_name', type=str, required=False, default="Selecticity-Experiments",
                        help='The name of the wandb project to save results in')
    config = parser.parse_args().config
    overrides = parser.parse_args().overrides
    print(f"\nUsing config {config}")
    print(f"\nOverrides: {overrides}")

    with open("configs/selectivity/" + config) as stream:
        try:
            base_config = yaml.safe_load(stream)
            base_config = override_config(base_config, overrides)
        except yaml.YAMLError as exc:
            raise RuntimeError(exc)

    ray.init(num_cpus=parser.parse_args().num_cpus, ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor
    if "wandb" in base_config and "api_key" in base_config["wandb"]:
        wandb.login(key=base_config["wandb"]["api_key"])

    tasks = []
    if parser.parse_args().ablation:
        # you can change the settings_options to run different ablation studies as you like
        # this will generate an outer product of all the hyperparameters
        settings_options = [
            ["d_state", [16]],
            ["seed", [0]],
            ["model.bias", [False]],
            ["model.B_is_selective", [True, False]],
            ["model.C_is_selective", [True, False]],
            ["model.dt_is_selective", [False, True]],
            ["model.channel_sharing", [False]],
            ["model.ssm_type", ["S6-Real", "S6-Complex"]],
        ]
    else:
        settings_options = []
    settings_options.append(['wandb.project', [parser.parse_args().project_name]])

    for config in experiments(settings_options):
        config.update({"comment": ""})
        config = override_config(base_config, [f"{k}={v}" for k, v in config.items()])
        print("\nCONFIG:")
        print(yaml.dump(config))
        tasks.append(run_experiment.remote(config, progress_bar_actor))
    pb.set_total(len(tasks))
    pb.print_until_done()

    print("finished running all")


if __name__ == "__main__":
    main()
