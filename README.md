# SSMComplexParamBenefits
Code for reproducing the experiments in the paper Provable Benefits of Complex Parameterizations for Structured State Space Models [TODO: add link to arxiv when we upload]

## Install Requirements
Tested with python 3.12.4.
```
pip install -r requirements.txt
```

## Experiments
### Theoretically Analyzed Settings
To run an experiment in the theoretically analyzed settings, use the following command:
```
python train_thoery.py --config <config_file>.yaml
```
where `<config_file>` is the name of the configuration file in the `configs` directory.
If you wish to run multiple experiments, you can modify the values in the main() function in train_theory.py.

### Real-World Setting
To reproduce the experiments in the real-world setting, 
please refer to the s4 repo from state-spaces [here](https://github.com/state-spaces/s4/).
After cloning the s4 repo, copy the configuration files into the relevant directory and run the following command:
```
python -m train experiment=lra/s4d-real-cifar 
```


### Selectivity
To reproduce the experiments on selectivity, use the following command:
```
python train.py --config <config_file>.yaml
```
where `<config_file>` is the name of the configuration file in the `configs` directory.
You can also modify the values in the main() function in train.py, to run a grid search over the hyperparameters.

## Citation
For citing the paper you can use:
```
@article{TODO,
  title={Provable Benefits of Complex Parameterizations for Structured State Space Models},
  author={TODO},
  journal={TODO},
  year={TODO}
}
```
