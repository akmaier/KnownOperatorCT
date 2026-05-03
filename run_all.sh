#!/usr/bin/env bash
# Single entry point invoked by the cloud agent.
#
# Each step is run in its own subshell so that a failure of one step does not
# stop the others. All stdout/stderr is tee'd into results/run_all.log so that
# the harvest script can attach failure traces to RESULTS.md.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p results
LOG="results/run_all.log"
: > "$LOG"

run_step() {
    local label="$1"
    shift
    local start_time
    start_time=$(date +%s)
    {
        echo "===== START: $label ====="
        echo "command: $*"
        echo "started: $(date -Iseconds)"
    } | tee -a "$LOG"
    if "$@" 2>&1 | tee -a "$LOG"; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        {
            echo "ok: $label (${elapsed}s)"
            echo "===== END: $label ====="
        } | tee -a "$LOG"
        echo "$label,ok,${elapsed}" >> results/run_all_steps.csv
    else
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        {
            echo "fail: $label (${elapsed}s)"
            echo "===== END: $label ====="
        } | tee -a "$LOG"
        echo "$label,fail,${elapsed}" >> results/run_all_steps.csv
    fi
}

if [[ ! -f results/run_all_steps.csv ]]; then
    echo "step,status,elapsed_s" > results/run_all_steps.csv
fi

run_step "surrogate" python src/run_surrogate.py --config configs/ct_surrogate.yaml
run_step "ct_train_known_operator" python src/ct_train.py --config configs/ct_full_resolution.yaml --model known_operator
run_step "ct_eval_known_operator" python src/ct_eval.py --config configs/ct_full_resolution.yaml --model known_operator
run_step "ct_train_fully_connected" python src/ct_train.py --config configs/ct_full_resolution.yaml --model fully_connected
run_step "ct_eval_fully_connected" python src/ct_eval.py --config configs/ct_full_resolution.yaml --model fully_connected
run_step "harvest" python src/harvest_results.py --output results/RESULTS.md
