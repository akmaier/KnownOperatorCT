#!/usr/bin/env bash
#SBATCH --job-name=ct_recons_var
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
# Need a 48 GB GPU. The Adagrad path's _multi_tensor implementation
# allocates multiple full-size weight-buffer intermediates during the step
# (~22 GB peak), so the 24 GB lme221 nodes OOM even after freeing KO.
# Pin to lme170 (RTX 8000 48 GB).
#SBATCH --nodelist=lme170
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme171,lme221,lme222,lme223
#SBATCH --output=results/slurm/recons-%x-%j.out
#SBATCH --error=results/slurm/recons-%x-%j.err
#
# One FC ablation at 256x256, N=1024, 1000 iters, plus the standard KO baseline.
#
# Pass the variant via --export. Required env vars:
#   VARIANT_TAG     short slug for the output filename, e.g. "adam-1e-3"
#   FC_OPTIMIZER    adagrad | adam
#   FC_LR           e.g. 1e-3
#   FC_BIAS         "yes" | "no"
#   FC_INIT         kaiming | xavier
#
# Submit with:
#   sbatch --export=ALL,VARIANT_TAG=adam-1e-3,FC_OPTIMIZER=adam,FC_LR=1e-3,FC_BIAS=no,FC_INIT=kaiming \
#          lme/slurm/recons_variant.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm "results/sample_efficiency_256/variants"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

: "${VARIANT_TAG:?must set VARIANT_TAG}"
: "${FC_OPTIMIZER:?must set FC_OPTIMIZER}"
: "${FC_LR:?must set FC_LR}"
: "${FC_BIAS:?must set FC_BIAS}"
: "${FC_INIT:?must set FC_INIT}"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID variant=$VARIANT_TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bias_flag=""
if [ "$FC_BIAS" = "yes" ]; then
    bias_flag="--fc-bias"
fi

no_relu_flag=""
if [ "${FC_NO_RELU:-no}" = "yes" ]; then
    no_relu_flag="--fc-no-relu"
fi

OUT="results/sample_efficiency_256/variants/recons_${VARIANT_TAG}.png"
TAG="opt=${FC_OPTIMIZER} lr=${FC_LR} bias=${FC_BIAS} init=${FC_INIT} no_relu=${FC_NO_RELU:-no}"

python src/render_reconstructions.py \
    --config configs/ct_sample_efficiency_256.yaml \
    --train-size 1024 --num-iterations 1000 --num-samples 2 \
    --out "$OUT" \
    --fc-optimizer "$FC_OPTIMIZER" \
    --fc-lr "$FC_LR" \
    $bias_flag \
    $no_relu_flag \
    --fc-init "$FC_INIT" \
    --fc-tag "$TAG"

echo "[$(date '+%F %T')] done -> $OUT"
