# DeepLight

## Installation
DeepLight requires `julia` and `python`. The install requirements are in `Project.toml` and `pyproject.toml` respectively.

## Usage
To configure the paths where the datasets and trained network weights will be stored, edit the file `run/paths.toml`.
Generating the training, validation and test data can be done by running the script `run/data_gen.jl`.
Once the data has been computed, `run/train_network.py` trains a set of 5 networks with and without Sobolev training.
