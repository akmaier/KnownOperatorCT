#!/usr/bin/env bash
# Helpers shared by the lab cluster training sbatch scripts.
#
# Provides:
#   forward_signals_to <pid>       — forward SIGTERM/SIGUSR1 from this script
#                                    to the python child, then keep waiting
#                                    until python actually exits.
#   maybe_resubmit <done_marker>   — if the .done sentinel does not exist and
#                                    the previous python exit code is 0, the
#                                    job hit its wall-time mid-training; we
#                                    resubmit ourselves with a dependency on
#                                    this job step. A run counter caps the
#                                    chain at 10 attempts so a real crash
#                                    can't loop forever.

set -uo pipefail

# Wait for $1 (PID), forwarding SIGTERM/SIGUSR1 from the sbatch wrapper to the
# child python process. Re-enter `wait` if it returned early because of a
# trapped signal (exit code > 128).
wait_with_signal_forwarding() {
    local pid="$1"
    trap "kill -TERM $pid 2>/dev/null || true" TERM
    trap "kill -USR1 $pid 2>/dev/null || true" USR1
    local rc=0
    while true; do
        wait "$pid"
        rc=$?
        if [ $rc -le 128 ]; then
            break
        fi
    done
    trap - TERM USR1
    return $rc
}

# Resubmit this same sbatch script if training did not complete. Pass the path
# to the per-model `.done` sentinel and the python exit code. We only resubmit
# on a clean exit (0) — non-zero is treated as a real crash.
maybe_resubmit() {
    local done_marker="$1"
    local prev_rc="$2"
    local max_chain="${RESUBMIT_MAX:-10}"
    local chain_idx="${CHAIN_IDX:-1}"

    if [ -f "$done_marker" ]; then
        echo "[wrapper] $done_marker present — training complete."
        return 0
    fi

    if [ "$prev_rc" -ne 0 ]; then
        echo "[wrapper] python exited with $prev_rc — not resubmitting (treat as real failure)."
        return "$prev_rc"
    fi

    if [ "$chain_idx" -ge "$max_chain" ]; then
        echo "[wrapper] reached max chain length $max_chain without completion — bailing out."
        return 1
    fi

    local next=$((chain_idx + 1))
    echo "[wrapper] training did not complete; submitting follow-up #$next ..."
    sbatch \
        --dependency="afterok:${SLURM_JOB_ID}" \
        --export="ALL,CHAIN_IDX=${next},RESUBMIT_MAX=${max_chain}" \
        "$0"
}
