import json

from roll.utils.tracing.core import TraceManager, TracingConfig
from roll.utils.tracing.exporter import export_trace_step


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


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


def test_export_trace_step_labels_throughput_series(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    manager = TraceManager(config=config, component="driver")

    manager.record_completed_span(
        "pipeline.step",
        phase="pipeline",
        category="pipeline",
        step=9,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        attrs={},
    )
    manager.record_sample(
        "vllm.prompt_throughput_tps",
        128.5,
        unit="tok/s",
        step=9,
        timestamp_ns=1_200_000_000,
        attrs={"source": "logger_manager", "aggregation": "sum"},
    )
    manager.record_sample(
        "vllm.generation_throughput_tps",
        64.0,
        unit="tok/s",
        step=9,
        timestamp_ns=1_300_000_000,
        attrs={"source": "logger_manager", "aggregation": "sum"},
    )
    manager.record_sample(
        "vllm.prompt_tokens_rate_tps",
        96.0,
        unit="tok/s",
        step=9,
        timestamp_ns=1_400_000_000,
        attrs={"source": "prometheus_counter", "aggregation": "sum"},
    )
    manager.flush(step=9)

    export_trace_step(str(tmp_path), step=9)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000009.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    series_by_name = {item["series_name"]: item for item in bundle["metric_series"]}
    assert series_by_name["vllm.prompt_throughput_tps"]["series_label"] == "Prompt 吞吐量"
    assert series_by_name["vllm.prompt_throughput_tps"]["series_group"] == "Throughput"
    assert series_by_name["vllm.generation_throughput_tps"]["series_label"] == "Generation 吞吐量"
    assert series_by_name["vllm.prompt_tokens_rate_tps"]["series_label"] == "Prompt Token Rate"


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


def test_export_trace_step_filters_metric_samples_to_latest_session(tmp_path):
    step_dir = tmp_path / "raw" / "steps" / "step_000001"
    sample_dir = tmp_path / "raw" / "samples" / "steps" / "step_000001"

    _write_jsonl(
        step_dir / "driver-pid1.jsonl",
        [
            {
                "trace_id": "old-trace",
                "span_id": "old-span",
                "parent_id": None,
                "name": "pipeline.step",
                "phase": "pipeline",
                "category": "pipeline",
                "step": 1,
                "sample_id": None,
                "traj_id": None,
                "attrs": {},
                "start_time_ns": 1_000_000_000,
                "end_time_ns": 2_000_000_000,
                "duration_ms": 1000.0,
                "pid": 1,
                "process_label": "driver",
            },
            {
                "trace_id": "new-trace",
                "span_id": "new-span",
                "parent_id": None,
                "name": "pipeline.step",
                "phase": "pipeline",
                "category": "pipeline",
                "step": 1,
                "sample_id": None,
                "traj_id": None,
                "attrs": {},
                "start_time_ns": 3_600_000_000_000,
                "end_time_ns": 3_601_000_000_000,
                "duration_ms": 1000.0,
                "pid": 1,
                "process_label": "driver",
            },
        ],
    )
    _write_jsonl(
        sample_dir / "worker_metrics-pid2.jsonl",
        [
            {
                "record_type": "sample",
                "name": "vllm.kv_blocks_used",
                "kind": "gauge",
                "unit": "blocks",
                "value": 10,
                "timestamp_ns": 1_500_000_000,
                "trace_id": None,
                "step": 1,
                "sample_id": None,
                "traj_id": None,
                "attrs": {"engine": "0"},
                "pid": 2,
                "process_label": "worker_metrics",
            },
            {
                "record_type": "sample",
                "name": "vllm.kv_blocks_used",
                "kind": "gauge",
                "unit": "blocks",
                "value": 20,
                "timestamp_ns": 3_600_500_000_000,
                "trace_id": None,
                "step": 1,
                "sample_id": None,
                "traj_id": None,
                "attrs": {"engine": "0"},
                "pid": 2,
                "process_label": "worker_metrics",
            },
        ],
    )

    export_trace_step(str(tmp_path), step=1)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000001.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    series = next(item for item in bundle["metric_series"] if item["series_name"] == "vllm.kv_blocks_used")
    assert bundle["summary"]["dropped_spans"] == 1
    assert bundle["summary"]["total_spans"] == 1
    assert bundle["summary"]["window_ms"] == 1000.0
    assert [point["value"] for point in series["points"]] == [20.0]
    assert [point["offset_ms"] for point in series["points"]] == [500.0]
    assert any("过滤掉 1 个不属于当前 session 的指标点" in note for note in bundle["notes"])


def test_export_trace_step_keeps_metric_series_without_spans(tmp_path):
    sample_dir = tmp_path / "raw" / "samples" / "steps" / "step_000001"
    _write_jsonl(
        sample_dir / "worker_metrics-pid2.jsonl",
        [
            {
                "record_type": "sample",
                "name": "vllm.kv_cache_usage_pct",
                "kind": "gauge",
                "unit": "%",
                "value": 42.0,
                "timestamp_ns": 1_000_000_000,
                "trace_id": None,
                "step": 1,
                "sample_id": None,
                "traj_id": None,
                "attrs": {"engine": "0"},
                "pid": 2,
                "process_label": "worker_metrics",
            },
        ],
    )

    export_trace_step(str(tmp_path), step=1)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000001.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert bundle["summary"]["total_spans"] == 0
    assert bundle["summary"]["window_ms"] == 0.0
    assert bundle["summary"]["processes"] == ["worker_metrics"]
    assert bundle["overview"]["process_count"] == 1
    assert bundle["metric_overview"] == {"series_count": 1, "sample_count": 1}
    assert bundle["metric_series"][0]["series_name"] == "vllm.kv_cache_usage_pct"
    assert bundle["metric_series"][0]["points"] == [
        {
            "offset_ms": 0.0,
            "value": 42.0,
            "attrs": {"engine": "0"},
            "sample_id": None,
            "traj_id": None,
        }
    ]


def test_export_trace_step_html_metric_labels_include_process_context(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    driver = TraceManager(config=config, component="driver")
    worker_a = TraceManager(config=config, component="worker_a")

    driver.record_completed_span(
        "pipeline.step",
        phase="pipeline",
        category="pipeline",
        step=9,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        attrs={},
    )
    worker_a.record_sample(
        "vllm.kv_blocks_used",
        128,
        unit="blocks",
        step=9,
        timestamp_ns=1_100_000_000,
        attrs={"engine": "0"},
    )
    driver.flush(step=9)
    worker_a.flush(step=9)

    html_path = export_trace_step(str(tmp_path), step=9)
    html = (tmp_path / "timeline" / "steps" / "step_000009.html").read_text(encoding="utf-8")

    assert html_path.endswith("step_000009.html")
    assert "metricSeriesContext(series)" in html
    assert "series.series_label + ' · ' + (series.process_label || 'process')" in html


def test_export_trace_step_uses_disambiguated_kv_capacity_labels(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    driver = TraceManager(config=config, component="driver")

    driver.record_completed_span(
        "pipeline.step",
        phase="pipeline",
        category="pipeline",
        step=11,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        attrs={},
    )
    driver.record_sample(
        "vllm.kv_cache_total_bytes",
        4096,
        unit="bytes",
        step=11,
        timestamp_ns=1_100_000_000,
        attrs={"engine": "0"},
    )
    driver.record_sample(
        "vllm.kv_cache_allocated_bytes",
        4608,
        unit="bytes",
        step=11,
        timestamp_ns=1_150_000_000,
        attrs={"engine": "0"},
    )
    driver.record_sample(
        "vllm.kv_cache_reserved_bytes",
        512,
        unit="bytes",
        step=11,
        timestamp_ns=1_200_000_000,
        attrs={"engine": "0"},
    )
    driver.flush(step=11)

    export_trace_step(str(tmp_path), step=11)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000011.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    series_by_name = {item["series_name"]: item for item in bundle["metric_series"]}

    assert series_by_name["vllm.kv_cache_total_bytes"]["series_label"] == "KV 可调度总容量"
    assert series_by_name["vllm.kv_cache_allocated_bytes"]["series_label"] == "KV 张量总容量"
    assert series_by_name["vllm.kv_cache_reserved_bytes"]["series_label"] == "KV 保留/未调度容量"


def test_export_trace_step_derives_kv_metrics_from_same_snapshot(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    driver = TraceManager(config=config, component="driver")

    driver.record_completed_span(
        "pipeline.step",
        phase="pipeline",
        category="pipeline",
        step=13,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        attrs={},
    )
    timestamp_ns = 1_200_000_000
    raw_samples = [
        ("vllm.kv_blocks_total", 100, "blocks"),
        ("vllm.kv_cached_blocks", 40, "blocks"),
        ("vllm.kv_blocks_free_cached", 10, "blocks"),
        ("vllm.kv_blocks_free_uncached", 30, "blocks"),
        ("vllm.prefix_cache_queries_delta", 50, "tokens"),
        ("vllm.prefix_cache_hits_delta", 25, "tokens"),
        ("vllm.kv_cache_allocated_bytes", 1000, "bytes"),
        ("vllm.kv_cache_reserved_bytes", 100, "bytes"),
        ("vllm.kv_event_stored_blocks_delta", 8, "blocks"),
        ("vllm.kv_event_removed_blocks_delta", 3, "blocks"),
    ]
    for name, value, unit in raw_samples:
        driver.record_sample(
            name,
            value,
            unit=unit,
            step=13,
            timestamp_ns=timestamp_ns,
            attrs={"engine": "0"},
        )
    driver.flush(step=13)

    export_trace_step(str(tmp_path), step=13)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000013.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    series_by_name = {item["series_name"]: item for item in bundle["metric_series"]}

    assert series_by_name["vllm.prefix_cache_hit_rate_delta_pct"]["points"][0]["value"] == 50.0
    assert series_by_name["vllm.kv_cached_block_pct"]["points"][0]["value"] == 40.0
    assert series_by_name["vllm.kv_evictable_block_pct"]["points"][0]["value"] == 10.0
    assert series_by_name["vllm.kv_cold_free_block_pct"]["points"][0]["value"] == 30.0
    assert series_by_name["vllm.kv_reserved_capacity_pct"]["points"][0]["value"] == 10.0
    assert series_by_name["vllm.kv_event_net_cached_blocks_delta"]["points"][0]["value"] == 5.0
    assert bundle["metric_overview"]["sample_count"] == len(raw_samples) + 6
    assert any("同一时间戳的 KV snapshot 派生出来的" in note for note in bundle["notes"])


def test_export_trace_step_includes_kv_usage_breakdown_series_labels(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    manager = TraceManager(config=config, component="driver")

    manager.record_completed_span(
        "pipeline.step",
        phase="pipeline",
        category="pipeline",
        step=14,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        attrs={},
    )
    manager.record_sample(
        "vllm.kv_cache_cached_free_usage_pct",
        12.5,
        unit="%",
        step=14,
        timestamp_ns=1_150_000_000,
        attrs={"engine": "0"},
    )
    manager.record_sample(
        "vllm.kv_cache_resident_usage_pct",
        62.5,
        unit="%",
        step=14,
        timestamp_ns=1_150_000_000,
        attrs={"engine": "0"},
    )
    manager.flush(step=14)

    export_trace_step(str(tmp_path), step=14)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000014.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    series_by_name = {item["series_name"]: item for item in bundle["metric_series"]}

    assert series_by_name["vllm.kv_cache_cached_free_usage_pct"]["series_label"] == "可驱逐 KV 使用率"
    assert series_by_name["vllm.kv_cache_resident_usage_pct"]["series_label"] == "KV 驻留使用率"
    assert series_by_name["vllm.kv_cache_cached_free_usage_pct"]["points"][0]["value"] == 12.5
    assert series_by_name["vllm.kv_cache_resident_usage_pct"]["points"][0]["value"] == 62.5


def test_export_trace_step_html_uses_shared_time_axis_layout(tmp_path):
    config = TracingConfig(enable=True, output_dir=str(tmp_path), timestamp_output_dir=False)
    driver = TraceManager(config=config, component="driver")

    driver.record_completed_span(
        "pipeline.step",
        phase="pipeline",
        category="pipeline",
        step=15,
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        attrs={},
    )
    driver.record_sample(
        "vllm.kv_blocks_total",
        128,
        unit="blocks",
        step=15,
        timestamp_ns=1_200_000_000,
        attrs={"engine": "0"},
    )
    driver.flush(step=15)

    export_trace_step(str(tmp_path), step=15)

    html = (tmp_path / "timeline" / "steps" / "step_000015.html").read_text(encoding="utf-8")

    assert 'id="timeline-viewport"' in html
    assert "const TIME_AXIS_LEFT_PAD = 330;" in html
    assert "function getTimeAxisLayout()" in html
    assert "function mirrorViewportScroll(source, target)" in html
    assert "metricViewport.addEventListener('scroll', () => mirrorViewportScroll(metricViewport, timelineViewport)" in html
    assert "timelineViewport.addEventListener('scroll', () => mirrorViewportScroll(timelineViewport, metricViewport)" in html


def test_export_trace_step_includes_overlapping_misc_spans_without_unpack_error(tmp_path):
    step_dir = tmp_path / "raw" / "steps" / "step_000001"
    misc_dir = tmp_path / "raw" / "misc"

    _write_jsonl(
        step_dir / "driver-pid1.jsonl",
        [
            {
                "trace_id": "trace-1",
                "span_id": "step-span",
                "parent_id": None,
                "name": "pipeline.step",
                "phase": "pipeline",
                "category": "pipeline",
                "step": 1,
                "sample_id": None,
                "traj_id": None,
                "attrs": {},
                "start_time_ns": 1_000_000_000,
                "end_time_ns": 2_000_000_000,
                "duration_ms": 1000.0,
                "pid": 1,
                "process_label": "driver",
            },
        ],
    )
    _write_jsonl(
        misc_dir / "env-pid9.jsonl",
        [
            {
                "trace_id": "trace-misc",
                "span_id": "misc-span",
                "parent_id": None,
                "name": "env.step",
                "phase": "env",
                "category": "interaction",
                "step": None,
                "sample_id": None,
                "traj_id": "traj-1",
                "attrs": {"env_id": 0},
                "start_time_ns": 1_100_000_000,
                "end_time_ns": 1_300_000_000,
                "duration_ms": 200.0,
                "pid": 9,
                "process_label": "env_manager",
            },
        ],
    )

    export_trace_step(str(tmp_path), step=1)

    bundle_path = tmp_path / "timeline" / "steps" / "step_000001.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert bundle["summary"]["total_spans"] == 2
    assert "interaction" in bundle["summary"]["categories"]
