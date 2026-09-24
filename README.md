# tpec2027_power_grid

## Introduction

Conference code for informative power-system probing through sequential Bayesian optimal experimental design. It compares Random, Fixed, Myopic, DAD, RL-sBOED, and Step-DAD on IEEE9 and IEEE14, with SIR-ODE as a supplementary benchmark.

The repository includes the [conference paper](TPEC_conference/output/pdf/eig_power_grid_conference.pdf) and [complete results and checkpoints](TPEC_conference/results/README.md).

## Installation

Requires Linux, Python 3.10+, an NVIDIA GPU, and a compatible CUDA toolkit with `nvcc` on `PATH`.

```bash
git clone https://github.com/alexschmidt123/tpec2027_power_grid.git
cd tpec2027_power_grid
python3 -m venv .venv
source .venv/bin/activate
# Install a CUDA-enabled PyTorch build compatible with your GPU here.
pip install -r requirements.txt
```

## Run

Run a small smoke test:

```bash
bash run.sh --config configs/ieee9_eig.yaml --smoke
bash run.sh --config configs/ieee14_eig.yaml --smoke
bash run.sh --config configs/sir_ode_eig.yaml --eval-seeds 1001 --smoke
```

Run one grid conference configuration:

```bash
bash run.sh --config configs/ieee14_eig.yaml --T 3 --seed 101 \
  --eval-seeds 1001,1002,1003 --contrasts 128 --eval-systems 512
```

Change the config, `--T`, or `--seed` as needed; use `--methods dad,random` to select methods. Outputs are saved under `experiments/`. New grid runs default to 1024 contrasts; the archived conference results use 128.
