#!/bin/bash
#SBATCH --job-name=tpec-a100-1001
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/user/g.lin/tpec_a100_timing_20261001/test1001-%j.log
set -euo pipefail
source /scratch/user/g.lin/Documents/objective_power_grid/hprc/environment.sh
cd /scratch/user/g.lin/tpec_a100_timing_20261001
for horizon in 3 4 5; do
  python /scratch/user/g.lin/profile_ieee9_a100_test1001.py --repo "$PWD" --T "$horizon" --systems 512 --eval-seed 1001 --warmup-seed 91001 --output "$PWD/results_test1001_step4/T$horizon"
done
