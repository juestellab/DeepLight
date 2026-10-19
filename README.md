# DeepLight

Code to reproduce the central results of the DeepLight paper: dataset generation,
training of the networks, scoring on the test set, and the paper's figures and
tables.

## Installation
DeepLight requires `julia` and `python` (>= 3.12). The requirements are in
`Project.toml`/`run/Project.toml` and `pyproject.toml` respectively. The light
transport is simulated with [MoCA](https://github.com/juestellab/MoCA), which has
to be available to the Julia environment.

## Usage
Set the locations of the dataset, the trained networks and the results in
`run/paths.toml`, then run
```
run/pipeline.sh
```
It has three stages, each of which can be run on its own with `--stages`, and
each of which skips or resumes work that is already done:

1. **dataset**: generates the training, validation and test sets
   (`run/generate_tissue.py`, `run/generate_light_transport.jl`). See the header
   of `run/pipeline.sh` for splitting the work over several GPUs or machines.
2. **train**: trains every network of the paper into `network_weights_dir`.
3. **analysis**: scores the networks on the test set into `results_dir` and draws
   the figures and tables into `results_dir/figures`.

`run/pipeline.sh --dry-run` reports what would be done.

## Training
| Runs | Script |
|---|---|
| main sweep: all direction arms x Sobolev weights, 2 seeds | `train_network.py sweep` |
| 5 seeds of the baseline and the DeepLight network | `train_network.py seeds` |
| equal simulation budget | `train_network.py size` |
| depth-channel ablation | `train_network.py depth_ablation` |
| loss transform of the operator (Fig. 4a) | `train_nl_modes.py` |
| loss transform of the derivative (Fig. 4b) | `train_nl_modes_dE.py` |

The run groups of `train_network.py` are defined in its `GROUPS`. Each run
directory is named after its configuration, including a digest of it, which is
how the analysis finds the runs again.

## Figures and tables
| Output | Script |
|---|---|
| Fig. 2 (loss transforms) | `nl_plot.py` |
| Fig. 3 (dataset samples) | `samples_figure.py` |
| Figs. 4, 5, 7, 8 (loss transform, quantitative results, depth channel, seeds and checkpoints) | `results_figure.py` |
| Tables 1 and 2 | `network_performance_table.py` |
| Figs. 6, 9 (qualitative samples) | `qualitative_samples.py` |

The MC noise analysis is not part of this release, so the noise line of Fig. 4
is not drawn.
