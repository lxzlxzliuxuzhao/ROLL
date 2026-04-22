#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$PWD:$PYTHONPATH"
export ROLL_TRACE_ENABLE=1
export ROLL_TRACE_DIR=./output/traces
export ROLL_TRACE_EXPORT_HTML=1
python examples/start_traced_agentic_pipeline.py --config_path $CONFIG_PATH --config_name agent_rollout_rock_swe_traced
