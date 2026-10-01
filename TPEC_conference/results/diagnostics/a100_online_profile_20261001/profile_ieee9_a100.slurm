#!/bin/bash
#SBATCH --job-name=tpec-a100-time
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/user/g.lin/tpec_a100_timing_20261001/job-%j.log
set -euo pipefail
source /scratch/user/g.lin/Documents/objective_power_grid/hprc/environment.sh
cd /scratch/user/g.lin/tpec_a100_timing_20261001
for horizon in 3 4 5; do
  python /scratch/user/g.lin/profile_ieee9_a100.py --repo "$PWD" --T "$horizon" --systems 32 --output "$PWD/results_final_step4/T$horizon"
done
