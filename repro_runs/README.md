# Repro Runs Layout

This folder is organized for paper-facing reproducibility.

- `waterbirds/train`: fixed training/repro runners.
- `waterbirds/sweeps`: hyperparameter sweep runners.
- `waterbirds/baselines`: zero-shot / non-training baselines.
- `redmeat/train`: fixed training/repro runners.
- `redmeat/sweeps`: hyperparameter sweep runners.
- `redmeat/baselines`: zero-shot / non-training baselines.
- `decoymnist/train`: fixed training/repro runners.
- `decoymnist/baselines`: zero-shot / non-training baselines.
- `third_party`: external dependency code kept vendored and untouched (`GALS`, `MakeMNIST`, `afr`, `group_DRO`).

Path handling in runners is now anchored to `repro_runs/third_party/*` so scripts remain runnable after this reorganization.
