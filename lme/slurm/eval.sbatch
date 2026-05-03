#!/usr/bin/env bash
#SBATCH --job-name=ct_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=lme49,lme171
#SBATCH --output=results/slurm/eval-%j.out
#SBATCH --error=results/slurm/eval-%j.err
#
# Evaluate one model. Pass model + config via env:
#   MODEL=known_operator   CONFIG=configs/ct_full_resolution.yaml
#   MODEL=fully_connected  CONFIG=configs/ct_fc_lme.yaml
#
# Submit from the repo root:
#   sbatch --export=ALL,MODEL=known_operator,CONFIG=configs/ct_full_resolution.yaml \
#          lme/slurm/eval.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm
source .venv/bin/activate

: "${MODEL:?must set MODEL=known_operator|fully_connected}"
: "${CONFIG:?must set CONFIG=path/to/cfg.yaml}"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID model=$MODEL"
python src/ct_eval.py --config "$CONFIG" --model "$MODEL"
echo "[$(date '+%F %T')] eval done"
