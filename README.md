# C2L
This repository is set up to develop and evaluate ML-based camera-to-lidar (C2L) calibration methods as well as general visual localization methods.

## Installation
Conda is used to create an environment with python 3.10 and pip is used to install the required python packages.
```
conda create --name c2l python=3.10
pip install -r requirements.txt
``` 

## Development Tools
[pre-commit](https://pre-commit.com/) is used to run code formatting and linting before each commit. The pre-commit configuration is stored in the .pre-commit-config.yaml file. autopep8 and pylint are used for formatting and linting python files respectively. [pre-commit-hook-yamlfmt](https://github.com/jumanjihouse/pre-commit-hook-yamlfmt) and [yamllint](https://github.com/adrienverge/yamllint.git) are used for formatting and linting yaml files respectively. To install pre-commit run the following command:
```
pre-commit install
```
In order for pylint to find project level imports, the project root directory must be added to the python path. This can be done by e.g. adding the following line to the .bashrc file:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/project/root"
```
