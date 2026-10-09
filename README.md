# Adaptive Policy-Based Power-System Probing

## Introduction

This project compares DAD, RL-sBOED, and Step-DAD with Random, Fixed, and Myopic for informative sequential probing on reduced IEEE 9-bus and IEEE 14-bus systems. The repository includes the [manuscript](TPEC_conference/eig_power_grid_conference.tex) and [experimental results and checkpoints](experiments/tpec2027_conference/README.md).

## Installation

Requires Conda, Linux, an NVIDIA GPU, and a compatible CUDA toolkit with `nvcc` available on `PATH`.

```bash
git clone https://github.com/alexschmidt123/tpec2027_power_grid.git
cd tpec2027_power_grid
conda create -n tpec2027 python=3.11 pip -y
conda activate tpec2027
python -m pip install -r requirements.txt
```

## Run

Activate the Conda environment and run a small smoke test:

```bash
conda activate tpec2027
bash run.sh --config configs/ieee9_eig.yaml --smoke
```

Run an experiment with 128 contrasts and 512 sequences per test seed:

```bash
bash run.sh --config configs/ieee14_eig.yaml --T 3 --seed 101 \
  --eval-seeds 1001,1002,1003 --contrasts 128 --eval-systems 512
```

Use `configs/ieee9_eig.yaml` for IEEE 9-bus and set `--T` to 3, 4, or 5. Outputs are saved under `experiments/`.
