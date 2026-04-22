#!/bin/bash
set +x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_PATH=$(basename $(dirname $0))
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export ROLL_TRACE_ENABLE=1
export ROLL_TRACE_DIR=./output/traces
export ROLL_TRACE_EXPORT_HTML=1
python examples/start_traced_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agent_val_rock_swe_traced
