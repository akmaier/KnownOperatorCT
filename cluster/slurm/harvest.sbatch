#!/usr/bin/env bash
#SBATCH --job-name=ct_harvest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/harvest-%j.out
#SBATCH --error=results/slurm/harvest-%j.err
#
# Collect every metrics JSON into results/RESULTS.md. CPU-only.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm
source .venv/bin/activate

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
python src/harvest_results.py --output results/RESULTS.md
echo "[$(date '+%F %T')] harvest done -> results/RESULTS.md"
