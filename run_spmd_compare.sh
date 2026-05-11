#!/usr/bin/env bash
# Run the three Qwen3-14B decode scopes (before-SPMD vs after-SPMD) on a2a3
# via task-submit. Six runs total, submitted sequentially. Each task gets its
# own auto-allocated NPU card and unlimited max-time. Stdout + stderr from
# each task are streamed to ./logs/<label>.log relative to this script.
#
# IMPORTANT: must run from the pypto-lib root so the golden helper module
# (used by every kernel script via `from golden import RunConfig, run`)
# resolves correctly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Ordered (label, python script path relative to SCRIPT_DIR) pairs.
TARGETS=(
    "scope1_before:models/qwen3/14b/qwen3_14b_decode.py"
    "scope1_after:models/qwen3/14b/qwen3_14b_decode_spmd.py"
    "scope1_before:models/qwen3/14b/qwen3_14b_decode_scope1.py"
    "scope1_after:models/qwen3/14b/qwen3_14b_decode_scope1_spmd.py"
    "scope2_before:models/qwen3/14b/qwen3_14b_decode_scope2.py"
    "scope2_after:models/qwen3/14b/qwen3_14b_decode_scope2_spmd.py"
    "scope3_before:models/qwen3/14b/qwen3_14b_decode_scope3.py"
    "scope3_after:models/qwen3/14b/qwen3_14b_decode_scope3_spmd.py"
)

PLATFORM="a2a3"

declare -a SUMMARY

for entry in "${TARGETS[@]}"; do
    label="${entry%%:*}"
    pyfile="${entry##*:}"
    log_file="$LOG_DIR/${label}.log"

    if [[ ! -f "$SCRIPT_DIR/$pyfile" ]]; then
        echo "[skip] $label: $pyfile not found" | tee -a "$log_file"
        SUMMARY+=("$label: SKIP (missing $pyfile)")
        continue
    fi

    echo "==== [$label] (cwd=$SCRIPT_DIR) python $pyfile -p $PLATFORM ===="
    start_ts=$(date +%s)

    if task-submit \
            --device auto \
            --max-time 0 \
            --run "cd $SCRIPT_DIR && python $pyfile -p $PLATFORM" \
            > >(tee "$log_file") 2>&1; then
        status="PASS"
    else
        status="FAIL"
    fi

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "==== [$label] $status (${elapsed}s) ===="
    SUMMARY+=("$label: $status (${elapsed}s)  log=$log_file")
done

echo
echo "================ SUMMARY ================"
printf '%s\n' "${SUMMARY[@]}"
