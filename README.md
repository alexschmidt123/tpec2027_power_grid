# Adaptive Policy-Based Power-System Probing

Code and archived results for **Adaptive Policy-Based Methods for Informative Power-System Probing**, prepared for IEEE TPEC 2027. The study compares Random, Fixed, Myopic, DAD, RL-sBOED, and Step-DAD on reduced IEEE 9-bus and IEEE 14-bus swing models. Information scores estimate the sequential prior-contrastive (sPCE) lower bound; they are not parameter-estimation accuracy or demonstrated control benefits.

The reviewed [manuscript source](TPEC_conference/eig_power_grid_conference.tex) is included. The [complete grid results](results/README.md) include raw evaluations, resolved configurations, timing measurements, larger-contrast rescoring, and trained checkpoints. Checkpoints are indexed under [IEEE 9-bus models](results/ieee9/03_models/) and [IEEE 14-bus models](results/ieee14/03_models/).

## Installation and execution

Use Linux, Python 3.10+, an NVIDIA GPU, a compatible CUDA toolkit with `nvcc` on `PATH`, and a CUDA-enabled PyTorch build. Install PyTorch for your environment, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install a compatible CUDA-enabled PyTorch build in this environment.
pip install -r requirements.txt
bash run.sh --config configs/ieee9_eig.yaml --smoke
```

To request a grid run with the archived conference contrast count:

```bash
bash run.sh --config configs/ieee14_eig.yaml --T 3 --seed 101 \
  --eval-seeds 1001,1002,1003 --contrasts 128 --eval-systems 512
```

This command does not guarantee identical paper results under a different source version or hardware; the original resolved run settings and results are included under `results/`. Training seeds are 101/202/303, test seeds are 1001/1002/1003, and each test seed has 512 sequences. Step-DAD uses four refinement updates. Timing results use A100 hardware, training seed 101, and test seed 1001.
