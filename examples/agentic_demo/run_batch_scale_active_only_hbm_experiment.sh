#!/bin/bash
set -euo pipefail
set +x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_PATH=$(basename "$(dirname "$0")")
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export ROLL_TRACE_ENABLE=${ROLL_TRACE_ENABLE:-1}
export ROLL_TRACE_EXPORT_HTML=${ROLL_TRACE_EXPORT_HTML:-1}
export ROLL_TRACE_EXPORT_STEP_INTERVAL=${ROLL_TRACE_EXPORT_STEP_INTERVAL:-1}
export MASTER_PORT=${MASTER_PORT:-6780}

BATCH_SIZES=${BATCH_SIZES:-"16 32 64 96"}
MAX_STEPS=${MAX_STEPS:-1}
RUN_ROOT=${RUN_ROOT:-"$REPO_ROOT/output/batch_scale_active_only_hbm_$(date +%Y%m%d_%H%M%S)"}
PYTHON_BIN=${PYTHON_BIN:-"/home/lxz/miniconda3/envs/broll/bin/python"}
RAY_BIN=${RAY_BIN:-"/home/lxz/miniconda3/envs/broll/bin/ray"}
ROCK_SSH_TARGET=${ROCK_SSH_TARGET:-"liuxuzhao@10.212.70.196"}
ROCK_TMUX_SESSION=${ROCK_TMUX_SESSION:-"rock_server"}
ROCK_RESET_BEFORE_SWEEP=${ROCK_RESET_BEFORE_SWEEP:-1}
ROCK_RESET_AFTER_RUN=${ROCK_RESET_AFTER_RUN:-1}
ROCK_RESET_WAIT_SECONDS=${ROCK_RESET_WAIT_SECONDS:-30}
RAY_STOP_BEFORE_RUN=${RAY_STOP_BEFORE_RUN:-1}
RAY_STOP_AFTER_RUN=${RAY_STOP_AFTER_RUN:-1}

EXPERIMENTS=(
  "lmcache_baseline:agent_rollout_rock_swe_traced_lmcache_baseline"
  "active_only_hbm:agent_rollout_rock_swe_traced_active_only_hbm"
  "no_lmcache:agent_rollout_rock_swe_traced_no_lmcache_sweep"
)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/traces" "$RUN_ROOT/rollout_dump"

reset_rock_service() {
  if [[ "$ROCK_RESET_AFTER_RUN" != "1" ]]; then
    return 0
  fi

  echo "[$(date '+%F %T')] Resetting ROCK service via ${ROCK_SSH_TARGET} tmux=${ROCK_TMUX_SESSION}"
  ssh "$ROCK_SSH_TARGET" \
    "tmux send-keys -t '$ROCK_TMUX_SESSION' 'rock admin stop' C-m; sleep 5; tmux send-keys -t '$ROCK_TMUX_SESSION' 'rock admin start' C-m"

  if [[ "$ROCK_RESET_WAIT_SECONDS" -gt 0 ]]; then
    echo "[$(date '+%F %T')] Waiting ${ROCK_RESET_WAIT_SECONDS}s for ROCK service restart"
    sleep "$ROCK_RESET_WAIT_SECONDS"
  fi
}

stop_ray_if_requested() {
  local when=$1
  local enabled=0
  if [[ "$when" == "before" ]]; then
    enabled=$RAY_STOP_BEFORE_RUN
  else
    enabled=$RAY_STOP_AFTER_RUN
  fi
  if [[ "$enabled" != "1" ]]; then
    return 0
  fi

  echo "[$(date '+%F %T')] ray stop --force (${when} run)"
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
}

MANIFEST="$RUN_ROOT/manifest.tsv"
printf 'experiment\tbatch_size\tconfig_name\tlog_path\ttrace_dir\trollout_dump_dir\tstatus\n' > "$MANIFEST"

if [[ "$ROCK_RESET_BEFORE_SWEEP" == "1" ]]; then
  if ! reset_rock_service; then
    echo "[$(date '+%F %T')] Initial ROCK reset failed; aborting sweep" >&2
    exit 1
  fi
fi

for item in "${EXPERIMENTS[@]}"; do
  experiment=${item%%:*}
  config_name=${item#*:}
  for batch_size in $BATCH_SIZES; do
    run_name="${experiment}_bs${batch_size}"
    log_path="$RUN_ROOT/logs/${run_name}.log"
    trace_dir="$RUN_ROOT/traces/${run_name}"
    rollout_dump_dir="$RUN_ROOT/rollout_dump/${run_name}"
    mkdir -p "$trace_dir" "$rollout_dump_dir"

    echo "[$(date '+%F %T')] START experiment=${experiment} batch_size=${batch_size} log=${log_path}"
    stop_ray_if_requested before
    status="ok"
    if ! ROLL_TRACE_DIR="$trace_dir" ROLL_TRACE_TIMESTAMP_OUTPUT_DIR=0 ROLL_TRACE_TIMESTAMP= \
      "$PYTHON_BIN" examples/start_traced_agentic_pipeline.py \
        --config_path "$CONFIG_PATH" \
        --config_name "$config_name" \
        max_steps="$MAX_STEPS" \
        rollout_batch_size="$batch_size" \
        train_env_manager.group_size="$batch_size" \
        train_env_manager.num_groups_partition="[1]" \
        tracing.output_dir="$trace_dir" \
        tracing.timestamp_output_dir=false \
        output_dir="$RUN_ROOT/output/${run_name}" \
        logging_dir="$RUN_ROOT/roll_logs/${run_name}" \
        rollout_dump_dir="$rollout_dump_dir" \
        model_name="$run_name" \
        > "$log_path" 2>&1; then
      status="failed"
      echo "[$(date '+%F %T')] FAILED experiment=${experiment} batch_size=${batch_size}; continuing"
    else
      echo "[$(date '+%F %T')] DONE experiment=${experiment} batch_size=${batch_size}"
    fi

    stop_ray_if_requested after

    if ! reset_rock_service; then
      status="${status}+rock_reset_failed"
      echo "[$(date '+%F %T')] ROCK reset failed after experiment=${experiment} batch_size=${batch_size}; continuing"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$experiment" "$batch_size" "$config_name" "$log_path" "$trace_dir" "$rollout_dump_dir" "$status" >> "$MANIFEST"
  done
done

"$PYTHON_BIN" examples/agentic_demo/summarize_batch_scale_active_only_hbm.py "$MANIFEST" \
  --output-csv "$RUN_ROOT/summary.csv" \
  --output-md "$RUN_ROOT/summary.md"

echo "Summary CSV: $RUN_ROOT/summary.csv"
echo "Summary MD:  $RUN_ROOT/summary.md"
