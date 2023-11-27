# C2L
This repository is set up to develop and evaluate ML-based camera-to-lidar (C2L) calibration methods as well as general visual localization methods.

## TODO List
- [ ] Integrate LOFTR model, a transformer based image correspondence finding network, from [LOFTR](https://zju3dv.github.io/loftr/) into the project
- [ ] Implement reprojection error and matching confidence based loss
- [ ] Implement optimizer instantiation from config
- [ ] Integrate logging into project
- [ ] Implement lightning module and instantiate from config
- And more...

## Installation
Conda is used to create an environment with python 3.10 and pip is used to install the required python packages.
```
conda create --name c2l python=3.10
pip install -r requirements.txt
``` 

## Experiments
The experiments are run using the [hydra](https://hydra.cc/) framework. Experiment configurations yaml files in *c2l/conf/experiment/* overwrite the default configuration in *c2l/conf/main_config.yaml*. Experiments can be run using the following command:
```
python c2l/main.py +experiment=my_config
```

## Development Tools
[pre-commit](https://pre-commit.com/) is used to run code formatting and linting before each commit. The pre-commit configuration is stored in the .pre-commit-config.yaml file. autopep8 and pylint are used for formatting and linting python files respectively. [yamllint](https://github.com/adrienverge/yamllint.git) is used for linting yaml files. To install pre-commit run the following command:
```
pre-commit install
```
In order for pylint to find project level imports, the project root directory must be added to the python path. This can be done by e.g. adding the following line to the .bashrc file:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/project/root"
```
