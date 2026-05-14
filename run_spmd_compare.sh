#!/usr/bin/env bash
# Run the three Qwen3-14B decode scopes (before-SPMD vs after-SPMD) on a2a3
# via task-submit. Six runs total, submitted sequentially. Each task gets its
# own auto-allocated NPU card and unlimited max-time. Stdout + stderr from
# each task are streamed to ./logs/<label>.log relative to this script.
#
# IMPORTANT: must run from the pypto-lib root so the golden helper module
# (used by every kernel script via `from golden import RunConfig, run`)
# resolves correctly.
#
# PyPTO compile logging (see pypto ``ir/compile.py`` + ``compile_profiling.py`` +
# ``pass_context.cpp``):
#   - PYTHONUNBUFFERED=1 / python -u  — flush Python + C++ stderr promptly.
#   - PYPTO_COMPILE_PROFILING=1       — per-stage timings → report/pipeline_profile.{txt,json}
#   - PYPTO_WARNING_LEVEL=post_pipeline — diagnostics also at pipeline end (perf hints are
#     registered for PostPipeline in ``diagnostic_check_registry.cpp``).
#   - PYPTO_VERIFY_LEVEL=roundtrip    — optional; very verbose/slow (unset by default).
# After each run we cat the newest ``build_output/<Program>_<ts>/report/*`` and every
# ``passes_dump/*.log`` so pass-local warnings land in the same log as the job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Ordered (label, python script path relative to SCRIPT_DIR) pairs.
TARGETS=(
    "qwen32b:models/qwen3/32b/qwen3_32b_decode2.py"
    # "scope1_before:models/qwen3/14b/qwen3_14b_decode.py"
    # "scope1_after:models/qwen3/14b/qwen3_14b_decode_spmd.py"
    # "scope1_before:models/qwen3/14b/qwen3_14b_decode_scope1.py"
    # "scope1_after:models/qwen3/14b/qwen3_14b_decode_scope1_spmd.py"
    # "scope2_before:models/qwen3/14b/qwen3_14b_decode_scope2.py"
    # "scope2_after:models/qwen3/14b/qwen3_14b_decode_scope2_spmd.py"
    # "scope3_before:models/qwen3/14b/qwen3_14b_decode_scope3.py"
    # "scope3_after:models/qwen3/14b/qwen3_14b_decode_scope3_spmd.py"
)

PLATFORM="${PLATFORM:-a2a3}"

declare -a SUMMARY

cleanup_wrap() {
    if [[ -n "${_WRAP_SCRIPT:-}" && -f "${_WRAP_SCRIPT}" ]]; then
        rm -f "${_WRAP_SCRIPT}"
    fi
}
trap cleanup_wrap EXIT

for entry in "${TARGETS[@]}"; do
    label="${entry%%:*}"
    pyfile="${entry##*:}"
    log_file="$LOG_DIR/${label}.log"

    if [[ ! -f "$SCRIPT_DIR/$pyfile" ]]; then
        echo "[skip] $label: $pyfile not found" | tee -a "$log_file"
        SUMMARY+=("$label: SKIP (missing $pyfile)")
        continue
    fi

    _WRAP_SCRIPT="$(mktemp -p "$LOG_DIR" "run_${label}_XXXXXX.sh")"
    chmod +x "$_WRAP_SCRIPT"
    cat >"$_WRAP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1
export PYPTO_COMPILE_PROFILING="\${PYPTO_COMPILE_PROFILING:-1}"
export PYPTO_WARNING_LEVEL="\${PYPTO_WARNING_LEVEL:-post_pipeline}"

python -u "$pyfile" -p "$PLATFORM"
_pypto_ec=\$?

_newest=""
if [[ -d "$SCRIPT_DIR/build_output" ]]; then
  _newest=\$(find "$SCRIPT_DIR/build_output" -mindepth 1 -maxdepth 1 -type d -printf '%T@\\t%p\\n' 2>/dev/null | sort -nr | head -1 | cut -f2- || true)
fi

echo ""
echo "======== PyPTO: newest build_output ========="
echo "DIR=\${_newest:-<none>}"
if [[ -n "\${_newest}" && -d "\${_newest}/report" ]]; then
  echo ""
  echo "--- report/ (memory, perf_hints, compile profile, ...) ---"
  for _f in "\${_newest}/report/"*; do
    [[ -e "\${_f}" ]] || continue
    echo ""
    echo "----- FILE: \${_f} -----"
    cat "\${_f}"
  done
fi
if [[ -n "\${_newest}" && -d "\${_newest}/passes_dump" ]]; then
  echo ""
  echo "--- passes_dump/*.log (per-pass warning dumps) ---"
  shopt -s nullglob
  _logs=( "\${_newest}/passes_dump/"*.log )
  if (("\${#_logs[@]}")); then
    for _f in "\${_logs[@]}"; do
      echo ""
      echo "----- FILE: \${_f} -----"
      cat "\${_f}"
    done
  else
    echo "(no .log files under passes_dump/)"
  fi
fi
echo ""
echo "======== End PyPTO artifact dump ========="
exit \$_pypto_ec
EOF

    echo "==== [$label] (cwd=$SCRIPT_DIR) bash $_WRAP_SCRIPT (platform=$PLATFORM) ====" | tee -a "$log_file"
    start_ts=$(date +%s)

    if task-submit \
            --device auto \
            --max-time 0 \
            --run "bash '$_WRAP_SCRIPT'" \
            > >(tee -a "$log_file") 2>&1; then
        status="PASS"
    else
        status="FAIL"
    fi

    rm -f "$_WRAP_SCRIPT"
    _WRAP_SCRIPT=""

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "==== [$label] $status (${elapsed}s) ====" | tee -a "$log_file"
    SUMMARY+=("$label: $status (${elapsed}s)  log=$log_file")
done

echo
echo "================ SUMMARY ================"
printf '%s\n' "${SUMMARY[@]}"
