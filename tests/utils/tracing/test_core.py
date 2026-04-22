import os
import json
from pathlib import Path

from roll.utils.tracing.core import TraceManager, TracingConfig, flush_trace_managers, get_trace_manager


def test_resolve_output_dir_appends_timestamp_once(monkeypatch):
    monkeypatch.setenv("ROLL_TRACE_TIMESTAMP", "20260416_153012")

    config = TracingConfig(enable=True, output_dir="./output/traces", timestamp_output_dir=True)
    resolved_once = config.resolve_output_dir()
    resolved_twice = config.resolve_output_dir()

    assert Path(resolved_once).name == "traces_20260416_153012"
    assert resolved_twice == resolved_once


def test_apply_env_canonicalizes_trace_dir(monkeypatch):
    monkeypatch.setenv("ROLL_TRACE_TIMESTAMP", "20260416_153012")

    config = TracingConfig(enable=True, output_dir="./output/traces", timestamp_output_dir=True)
    config.apply_env()

    assert Path(os.environ["ROLL_TRACE_DIR"]).name == "traces_20260416_153012"
    assert os.environ["ROLL_TRACE_TIMESTAMP_OUTPUT_DIR"] == "1"


def test_resolve_output_dir_can_disable_timestamp(monkeypatch):
    monkeypatch.setenv("ROLL_TRACE_TIMESTAMP", "20260416_153012")

    config = TracingConfig(enable=True, output_dir="./output/traces", timestamp_output_dir=False)

    assert config.resolve_output_dir() == "./output/traces"


def test_trace_manager_flushes_metric_samples(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    manager = TraceManager(config=config, component="worker")

    manager.record_sample(
        "vllm.kv_cache_usage_pct",
        37.5,
        unit="%",
        step=3,
        timestamp_ns=123456789,
        attrs={"engine": "0"},
    )
    manager.flush(step=3)

    sample_path = tmp_path / "raw" / "samples" / "steps" / "step_000003" / f"worker-pid{os.getpid()}.jsonl"
    assert sample_path.exists()
    records = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["record_type"] == "sample"
    assert records[0]["name"] == "vllm.kv_cache_usage_pct"
    assert records[0]["value"] == 37.5
    assert records[0]["unit"] == "%"


def test_flush_trace_managers_flushes_existing_process_managers(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    manager = get_trace_manager(config=config, component="worker")

    manager.record_sample(
        "vllm.kv_blocks_used",
        12,
        unit="blocks",
        step=5,
        timestamp_ns=22334455,
        attrs={"engine": "0"},
    )
    flush_trace_managers(step=5)

    sample_path = tmp_path / "raw" / "samples" / "steps" / "step_000005" / f"worker-pid{os.getpid()}.jsonl"
    assert sample_path.exists()
    records = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["name"] == "vllm.kv_blocks_used"
    assert records[0]["value"] == 12
