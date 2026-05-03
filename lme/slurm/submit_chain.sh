#!/usr/bin/env bash
# Submit the full reproduction pipeline as a Slurm chain on the LME cluster.
#
#   surrogate (CPU)
#       │
#       ├── ko_train  (1 GPU; auto-resumes if hits 24h)
#       │       │
#       │       └── ko_eval (1 GPU)
#       │
#       └── fc_train  (1 big GPU OR 4-GPU FSDP depending on flag)
#               │
#               └── fc_eval (1 GPU)
#                       │
#                       └── harvest (CPU; depends on both eval jobs)
#
# Usage:
#   bash lme/slurm/submit_chain.sh             # FC on a single big GPU (default)
#   bash lme/slurm/submit_chain.sh --fsdp      # FC via 4-GPU FSDP
#
# Each training step's auto-resubmit is handled inside the train_*.sbatch
# script itself. The harvest job uses afterok on both eval jobs so it only
# runs once both finished cleanly.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

FC_MODE="single"
if [ "${1:-}" = "--fsdp" ]; then
    FC_MODE="fsdp"
fi

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
