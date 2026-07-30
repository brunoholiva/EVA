#!/bin/bash
#SBATCH --partition=SP2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#
# Usage:
#   sbatch run_eva.slurm /path/to/config.toml
#
# The config's [output] section determines where results land.


CONFIG="${1:-config.toml}"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Config:    $CONFIG"

micromamba run -n eva python EVA.py --config "$CONFIG"
