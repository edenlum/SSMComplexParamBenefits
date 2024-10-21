# SSMComplexParamBenefits
Code for reproducing the experiments in the paper Provable Benefits of Complex Parameterizations for Structured State Space Models [https://arxiv.org/abs/2410.14067]

## Setup

### Install Requirements

Tested with Python 3.12.4.

```bash
pip install -r requirements.txt
```

### Setup W&B

To use Weights & Biases (W&B) for tracking your experiments, you need to log in to your W&B account. You can do this by running the following command:

```bash
wandb login
```

When prompted, paste your API key, which you can find on your [W&B settings page](https://wandb.ai/settings) after logging in. This will store your credentials locally, allowing W&B to track and log your experiments automatically.

For more detailed instructions on setting up W&B, you can refer to the [W&B Quickstart Guide](https://docs.wandb.ai/quickstart)

## Experiments
### Running Theoretical SSM Experiments

To run a set of experiments in the theoretically analyzed settings, use the following command:

```bash
python train_theory.py --experiment <real,complex> --num_cpus <number_of_cpus> --project_name <your_project_name>
```

- `<experiment>`: Choose either `real` or `complex` to run a set of experiments.
- `<num_cpus>`: Specify the number of CPUs to use. Each experiment in the set will run on a different CPU. Defaults to 4.
- `<project_name>`: (Optional) Specify the name of the W&B project where results will be saved. Defaults to `TheoreticalSSM`.

**Results will be available on your W&B project page.**

For example:

```bash
python train_theory.py --experiment real --project_name MyProject --num_cpus 4
```

This will run the `real` experiment set using `4` CPUs, with each experiment in the set running on a separate CPU, and log the results under `MyProject` on W&B.

### Real-World Setting
To reproduce the experiments in the real-world setting, 
please refer to the s4 repo from state-spaces [here](https://github.com/state-spaces/s4/).
After cloning the s4 repo, copy the configuration files from `configs/real_world` into the `configs/experiment/lra` directory in the s4 repo and run the following command:
```
python -m train experiment=lra/<config_file>.yaml
```
where `<config_file>` is the name of the configuration file in the `configs/experiment/lra` directory (in the s4 repo).

### Selectivity

To reproduce a single run from the experiments on selectivity, use the following command:

```bash
python train_selectivity.py --config <config_file>.yaml --num_cpus <number_of_cpus> --project_name <your_project_name>
```

- `<config_file>`: The name of the configuration file in the `configs/selectivity` directory (either `delay.yaml` or `induction.yaml`).
- `<num_cpus>`: Specify the number of CPUs to use. Each experiment in the set will run on a different CPU. Defaults to 4.
- `<project_name>`: (Optional) Specify the name of the W&B project where results will be saved. Defaults to `Selectivity-Experiments`.

To run the comparison between complex and real models from the paper, add the `--compare_real_complex` flag.  This will run multiple experiments in parallel, each with different hyperparameters.

To run the ablation study from the paper, add the `--ablation` flag. This will run multiple experiments in parallel, each with different hyperparameters.

You can play with the hyperparameters in the 'main()' function in 'train_selectivity.py'.

**Example Usage:**

```bash
python train_selectivity.py --config induction.yaml --ablation
```

This command will use the configuration from `configs/selectivity/induction.yaml` and run the ablation study with 1 seed.

**Results will be available on your W&B project page.**

## Citation
For citing the paper you can use:
```
@misc{ranmilo2024provablebenefitscomplexparameterizations,
      title={Provable Benefits of Complex Parameterizations for Structured State Space Models}, 
      author={Yuval Ran-Milo and Eden Lumbroso and Edo Cohen-Karlik and Raja Giryes and Amir Globerson and Nadav Cohen},
      year={2024},
      eprint={2410.14067},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.14067}, 
}
```
