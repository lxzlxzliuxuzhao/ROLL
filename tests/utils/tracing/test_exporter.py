import json

from roll.utils.tracing.core import TraceManager, TracingConfig
from roll.utils.tracing.exporter import export_trace_step


def test_export_trace_step_includes_metric_series(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    manager = TraceManager(config=config, component="driver")

    manager.record_completed_span(
        "pipeline.step",
        phase="pipeline",
        category="pipeline",
        step=7,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        attrs={},
    )
    manager.record_completed_span(
        "inference.request",
        phase="inference",
        category="inference",
        step=7,
        sample_id="traj-1:step:0",
        traj_id="traj-1",
        start_time_ns=1_100_000_000,
        end_time_ns=1_400_000_000,
        attrs={"prompt_tokens": 128, "output_tokens": 32, "request_id": "req-1"},
    )
    manager.record_sample(
        "vllm.kv_cache_usage_pct",
        21.5,
        unit="%",
        step=7,
        timestamp_ns=1_150_000_000,
        attrs={"engine": "0"},
    )
    manager.record_sample(
        "vllm.kv_cache_usage_pct",
        46.0,
        unit="%",
        step=7,
        timestamp_ns=1_350_000_000,
        attrs={"engine": "0"},
    )
    manager.record_sample(
        "vllm.kv_blocks_used",
        320,
        unit="blocks",
        step=7,
        timestamp_ns=1_200_000_000,
        attrs={"engine": "0"},
    )
    manager.record_sample(
        "vllm.num_preemptions_delta",
        2,
        unit="count",
        step=7,
        timestamp_ns=1_250_000_000,
        attrs={"engine": "0"},
    )
    manager.flush(step=7)

    export_trace_step(str(tmp_path), step=7)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000007.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert bundle["metric_overview"]["series_count"] == 3
    assert bundle["metric_overview"]["sample_count"] == 4

    series_by_name = {item["series_name"]: item for item in bundle["metric_series"]}
    assert series_by_name["vllm.kv_cache_usage_pct"]["series_label"] == "KV Cache 使用率"
    assert series_by_name["vllm.kv_cache_usage_pct"]["engine_label"] == "0"
    assert [point["value"] for point in series_by_name["vllm.kv_cache_usage_pct"]["points"]] == [21.5, 46.0]
    assert series_by_name["vllm.kv_blocks_used"]["series_label"] == "Active Blocks"
    assert series_by_name["vllm.num_preemptions_delta"]["series_label"] == "抢占次数增量"


def test_export_trace_step_separates_same_engine_metrics_by_process(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    driver = TraceManager(config=config, component="driver")
    worker_a = TraceManager(config=config, component="worker_a")
    worker_b = TraceManager(config=config, component="worker_b")

    driver.record_completed_span(
        "pipeline.step",
        phase="pipeline",
        category="pipeline",
        step=3,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        attrs={},
    )
    worker_a.record_sample(
        "vllm.kv_blocks_used",
        128,
        unit="blocks",
        step=3,
        timestamp_ns=1_100_000_000,
        attrs={"engine": "0"},
    )
    worker_b.record_sample(
        "vllm.kv_blocks_used",
        256,
        unit="blocks",
        step=3,
        timestamp_ns=1_200_000_000,
        attrs={"engine": "0"},
    )
    driver.flush(step=3)
    worker_a.flush(step=3)
    worker_b.flush(step=3)

    export_trace_step(str(tmp_path), step=3)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000003.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    kv_series = [
        item for item in bundle["metric_series"] if item["series_name"] == "vllm.kv_blocks_used"
    ]
    assert len(kv_series) == 2
    assert {item["process_label"] for item in kv_series} == {"worker_a", "worker_b"}
    assert {item["series_key"] for item in kv_series} == {
        "vllm_kv_blocks_used:worker_a:0",
        "vllm_kv_blocks_used:worker_b:0",
    }
    assert {item["latest_value"] for item in kv_series} == {128.0, 256.0}
