#!/bin/bash
#SBATCH --job-name=pxm_sample
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/healthinfolab/phz24002/PocketXMol/logs/sample_%j.out
#SBATCH --error=/shared/healthinfolab/phz24002/PocketXMol/logs/sample_%j.err

# PocketXMol Sampling Script for SLURM
# Usage:
#   sbatch pocketxmol_sample_slurm.sh [config_task] [outdir]
# Examples:
#   sbatch pocketxmol_sample_slurm.sh \
#     configs/sample/test/sbdd_csd/base.yml \
#     outputs_test/sbdd_csd_base
#   sbatch pocketxmol_sample_slurm.sh \
#     configs/sample/test/sbdd_csd/simple.yml \
#     outputs_test/sbdd_csd_noAR
SCRIPT_DIR="/shared/healthinfolab/phz24002/PocketXMol"
cd "$SCRIPT_DIR" || exit 1
echo "PWD: $(pwd)"

CONFIG_TASK=${1:-configs/sample/test/sbdd_csd/simple.yml}
OUTDIR=${2:-/shared/healthinfolab/phz24002/PocketXMol/outputs_test/sbdd_csd_noAR}

set -euo pipefail

CONDA_ENV_PATH="/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$OUTDIR"

echo "=========================================="
echo "PocketXMol Sampling (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Config task: $CONFIG_TASK"
echo "Outdir: $OUTDIR"
echo "Started: $(date)"
echo "=========================================="

python scripts/sample_drug3d.py \
  --config_task "$CONFIG_TASK" \
  --outdir "$OUTDIR"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE

