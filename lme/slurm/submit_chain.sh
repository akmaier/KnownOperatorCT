#!/usr/bin/env bash
# Submit the full reproduction pipeline as a Slurm chain on the LME cluster.
#
#   surrogate (CPU)
#       │
#       ├── ko_train  (1 GPU; auto-resumes if hits 24h)
#       │       │
#       │       └── ko_eval (1 GPU)
#       │
#       └── fc_train  (4-GPU FSDP by default; --single forces single big-GPU)
#               │
#               └── fc_eval (1 GPU)
#                       │
#                       └── harvest (CPU; depends on both eval jobs)
#
# Usage:
#   bash lme/slurm/submit_chain.sh             # FC via 4-GPU FSDP (default)
#   bash lme/slurm/submit_chain.sh --single    # FC on a single 48 GB GPU (rare)
#
# Why FSDP is the default: the FC model + Adagrad state + gradients want
# ~28 GB at peak. The only LME nodes with enough headroom on a single GPU
# are the RTX 8000 / A6000 boxes (48 GB), which are usually busy or broken.
# FSDP across 4 small GPUs is the path that reliably finishes.
#
# Each training step's auto-resubmit is handled inside the train_*.sbatch
# script itself. The harvest job uses afterok on both eval jobs so it only
# runs once both finished cleanly.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Slurm opens the --output / --error paths before the sbatch script runs.
mkdir -p results/slurm results/checkpoints

FC_MODE="fsdp"
case "${1:-}" in
    --fsdp)   FC_MODE="fsdp" ;;
    --single) FC_MODE="single" ;;
    "")       ;;
    *) echo "unknown flag: $1 (expected --fsdp or --single)"; exit 2 ;;
esac

# Helper: extract the JobId from `sbatch` output ("Submitted batch job 12345").
submit() {
    local out
    out=$(sbatch "$@")
    echo "$out" >&2
    awk '{print $4}' <<<"$out"
}

KO_CFG="configs/ct_full_resolution.yaml"
FC_CFG="configs/ct_fc_lme.yaml"

echo "Submitting surrogate ..."
SURR_ID=$(submit lme/slurm/surrogate.sbatch)

echo "Submitting KO training (after surrogate) ..."
KO_TRAIN_ID=$(submit --dependency=afterok:"$SURR_ID" lme/slurm/train_ko.sbatch)

echo "Submitting KO eval (after KO training) ..."
KO_EVAL_ID=$(submit \
    --dependency=afterok:"$KO_TRAIN_ID" \
    --export="ALL,MODEL=known_operator,CONFIG=$KO_CFG" \
    lme/slurm/eval.sbatch)

if [ "$FC_MODE" = "fsdp" ]; then
    echo "Submitting FC FSDP training (after surrogate) ..."
    FC_TRAIN_ID=$(submit --dependency=afterok:"$SURR_ID" lme/slurm/train_fc_fsdp.sbatch)
else
    echo "Submitting FC single-GPU training (after surrogate) ..."
    echo "  WARNING: needs a 48 GB GPU; lme221's 24 GB OOMs in backward."
    FC_TRAIN_ID=$(submit --dependency=afterok:"$SURR_ID" lme/slurm/train_fc.sbatch)
fi

echo "Submitting FC eval (after FC training) ..."
FC_EVAL_ID=$(submit \
    --dependency=afterok:"$FC_TRAIN_ID" \
    --export="ALL,MODEL=fully_connected,CONFIG=$FC_CFG" \
    lme/slurm/eval.sbatch)

echo "Submitting harvest (after both evals) ..."
HARVEST_ID=$(submit \
    --dependency=afterok:"$KO_EVAL_ID":"$FC_EVAL_ID" \
    lme/slurm/harvest.sbatch)

echo
echo "Chain submitted:"
echo "  surrogate:     $SURR_ID"
echo "  ko_train:      $KO_TRAIN_ID"
echo "  ko_eval:       $KO_EVAL_ID"
echo "  fc_train ($FC_MODE): $FC_TRAIN_ID"
echo "  fc_eval:       $FC_EVAL_ID"
echo "  harvest:       $HARVEST_ID"
echo
echo "Watch the queue:  squeue -u \$(whoami)"
echo "Final report:     results/RESULTS.md"
