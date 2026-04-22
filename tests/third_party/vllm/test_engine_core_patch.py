from types import SimpleNamespace
from unittest.mock import patch

from roll.third_party.vllm.engine_core_patch import (
    _evict_free_cached_blocks,
    _sum_kv_cache_tensor_bytes,
    _sum_page_size_bytes,
    patch_vllm_engine_core_inspection,
)


def test_engine_core_patch_exposes_utility_on_engine_core_proc():
    import vllm.v1.engine.core as core

    patch_vllm_engine_core_inspection()

    assert hasattr(core.EngineCore, "roll_get_kv_cache_snapshot")
    assert hasattr(core.EngineCoreProc, "roll_get_kv_cache_snapshot")
    assert hasattr(core.EngineCore, "roll_reset_lmcache_engine")
    assert hasattr(core.EngineCoreProc, "roll_reset_lmcache_engine")
    assert hasattr(core.EngineCore, "roll_evict_cached_free_blocks")
    assert hasattr(core.EngineCoreProc, "roll_evict_cached_free_blocks")
    assert hasattr(core.EngineCoreProc, "_roll_original_run_engine_core")
    assert core.EngineCoreProc.run_engine_core is not core.EngineCoreProc._roll_original_run_engine_core


def test_evict_free_cached_blocks_respects_target():
    evicted = []

    def _maybe_evict(block):
        if block.block_hash is None:
            return False
        block.block_hash = None
        evicted.append(block.block_id)
        return True

    block_pool = SimpleNamespace(
        enable_caching=True,
        blocks=[
            SimpleNamespace(block_id=0, is_null=True, ref_cnt=0, block_hash=None),
            SimpleNamespace(block_id=1, is_null=False, ref_cnt=1, block_hash="active"),
            SimpleNamespace(block_id=2, is_null=False, ref_cnt=0, block_hash="cached-a"),
            SimpleNamespace(block_id=3, is_null=False, ref_cnt=0, block_hash="cached-b"),
            SimpleNamespace(block_id=4, is_null=False, ref_cnt=0, block_hash=None),
        ],
        _maybe_evict_cached_block=_maybe_evict,
    )

    summary = _evict_free_cached_blocks(block_pool, target_free_cached_blocks=0)

    assert summary == {
        "target_free_cached_blocks": 0,
        "free_cached_before": 2,
        "free_cached_after": 0,
        "evicted_blocks": 2,
    }
    assert evicted == [2, 3]


def test_sum_page_size_bytes_uses_memory_pool_count_for_uniform_page_sizes():
    kv_cache_manager = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=[
                SimpleNamespace(size=1024),
                SimpleNamespace(size=1024),
                SimpleNamespace(size=1024),
            ],
            kv_cache_groups=[
                SimpleNamespace(layer_names=["layer.0", "layer.1"], kv_cache_spec=SimpleNamespace(page_size_bytes=256)),
                SimpleNamespace(layer_names=["layer.2"], kv_cache_spec=SimpleNamespace(page_size_bytes=256)),
            ],
        )
    )

    assert _sum_page_size_bytes(kv_cache_manager) == 256 * 3


def test_sum_kv_cache_tensor_bytes_accumulates_allocated_tensor_size():
    kv_cache_manager = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=[
                SimpleNamespace(size=1024),
                SimpleNamespace(size=2048),
                SimpleNamespace(size=512),
            ],
            kv_cache_groups=[],
        )
    )

    assert _sum_kv_cache_tensor_bytes(kv_cache_manager) == 3584


def test_roll_get_kv_cache_snapshot_distinguishes_schedulable_and_allocated_bytes():
    import vllm.v1.engine.core as core

    patch_vllm_engine_core_inspection()

    active_block = SimpleNamespace(is_null=False, ref_cnt=1, block_hash="active")
    free_cached_block = SimpleNamespace(is_null=False, ref_cnt=0, block_hash="cached")
    free_uncached_block = SimpleNamespace(is_null=False, ref_cnt=0, block_hash=None)
    another_active_block = SimpleNamespace(is_null=False, ref_cnt=1, block_hash=None)
    null_block = SimpleNamespace(is_null=True, ref_cnt=0, block_hash=None)

    block_pool = SimpleNamespace(
        num_gpu_blocks=5,
        get_num_free_blocks=lambda: 2,
        cached_block_hash_to_block={"cached": {1: free_cached_block}, "active": {0: active_block}},
        blocks=[null_block, active_block, free_cached_block, free_uncached_block, another_active_block],
        enable_kv_cache_events=False,
    )
    kv_cache_manager = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=[SimpleNamespace(size=1280)],
            kv_cache_groups=[
                SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=SimpleNamespace(page_size_bytes=256))
            ],
        ),
        block_pool=block_pool,
        usage=0.5,
        num_kv_cache_groups=1,
        enable_caching=True,
    )
    scheduler = SimpleNamespace(kv_cache_manager=kv_cache_manager, running=[], waiting=[])
    engine = SimpleNamespace(scheduler=scheduler)

    snapshot = core.EngineCore.roll_get_kv_cache_snapshot(engine)

    assert snapshot["available"] is True
    assert snapshot["num_gpu_blocks_total"] == 4
    assert snapshot["bytes_per_block"] == 256
    assert snapshot["kv_cache_total_bytes"] == 1024
    assert snapshot["kv_cache_allocated_bytes"] == 1280
    assert snapshot["kv_cache_reserved_bytes"] == 256
    assert snapshot["kv_cache_used_bytes"] == 512
    assert snapshot["kv_cache_free_bytes"] == 512
    assert snapshot["kv_cache_active_usage_perc"] == 50.0
    assert snapshot["kv_cache_cached_free_usage_perc"] == 25.0
    assert snapshot["kv_cache_resident_usage_perc"] == 75.0


def test_roll_get_kv_cache_snapshot_prefers_raw_block_pool_scan_for_realtime_counts():
    import vllm.v1.engine.core as core

    patch_vllm_engine_core_inspection()

    active_block = SimpleNamespace(is_null=False, ref_cnt=1, block_hash="active")
    free_cached_block = SimpleNamespace(is_null=False, ref_cnt=0, block_hash="cached")
    free_uncached_block = SimpleNamespace(is_null=False, ref_cnt=0, block_hash=None)
    another_active_block = SimpleNamespace(is_null=False, ref_cnt=2, block_hash=None)
    null_block = SimpleNamespace(is_null=True, ref_cnt=0, block_hash=None)

    block_pool = SimpleNamespace(
        num_gpu_blocks=5,
        get_num_free_blocks=lambda: 0,
        get_num_cached_free_blocks=lambda: 0,
        cached_block_hash_to_block={"cached": {1: free_cached_block}, "active": {0: active_block}},
        blocks=[null_block, active_block, free_cached_block, free_uncached_block, another_active_block],
        enable_kv_cache_events=False,
    )
    kv_cache_manager = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=[SimpleNamespace(size=1280)],
            kv_cache_groups=[
                SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=SimpleNamespace(page_size_bytes=256))
            ],
        ),
        block_pool=block_pool,
        usage=0.0,
        num_kv_cache_groups=1,
        enable_caching=True,
    )
    scheduler = SimpleNamespace(kv_cache_manager=kv_cache_manager, running=[], waiting=[])
    engine = SimpleNamespace(scheduler=scheduler)

    snapshot = core.EngineCore.roll_get_kv_cache_snapshot(engine)

    assert snapshot["num_gpu_blocks_used"] == 2
    assert snapshot["num_gpu_blocks_free"] == 2
    assert snapshot["cached_block_count"] == 2
    assert snapshot["free_cached_block_count"] == 1
    assert snapshot["free_uncached_block_count"] == 1
    assert snapshot["free_cached_block_count_counter"] == 0
    assert snapshot["kv_cache_active_usage_perc"] == 50.0
    assert snapshot["kv_cache_cached_free_usage_perc"] == 25.0
    assert snapshot["kv_cache_resident_usage_perc"] == 75.0


def test_scheduler_make_stats_attaches_cached_free_and_resident_usage():
    import vllm.v1.core.sched.scheduler as scheduler_mod
    from vllm.v1.metrics.stats import PrefixCacheStats

    patch_vllm_engine_core_inspection()

    block_pool = SimpleNamespace(
        num_gpu_blocks=5,
        get_num_free_blocks=lambda: 2,
        get_num_cached_free_blocks=lambda: 1,
    )
    kv_cache_manager = SimpleNamespace(
        usage=0.5,
        make_prefix_cache_stats=lambda: PrefixCacheStats(),
        block_pool=block_pool,
    )
    scheduler = SimpleNamespace(
        log_stats=True,
        kv_cache_manager=kv_cache_manager,
        running=[],
        waiting=[],
    )

    stats = scheduler_mod.Scheduler.make_stats(scheduler)

    assert stats is not None
    assert stats.kv_cache_usage == 0.5
    assert stats.kv_cache_active_usage == 0.5
    assert stats.kv_cache_cached_free_usage == 0.25
    assert stats.kv_cache_resident_usage == 0.75


def test_logging_stat_logger_logs_kv_usage_breakdown():
    import vllm.v1.metrics.loggers as metric_loggers
    from vllm.v1.metrics.stats import SchedulerStats

    patch_vllm_engine_core_inspection()

    logging_stat_logger = metric_loggers.LoggingStatLogger(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks=8)),
        engine_index=2,
    )
    logging_stat_logger.last_prompt_throughput = 1.0
    logging_stat_logger.last_scheduler_stats = SchedulerStats(
        num_running_reqs=3,
        num_waiting_reqs=1,
        kv_cache_usage=0.5,
    )
    logging_stat_logger.last_scheduler_stats.kv_cache_active_usage = 0.5
    logging_stat_logger.last_scheduler_stats.kv_cache_cached_free_usage = 0.25
    logging_stat_logger.last_scheduler_stats.kv_cache_resident_usage = 0.75
    logging_stat_logger._roll_engine_core_snapshot = {
        "engine": "2",
        "num_requests_running": 0,
        "num_requests_waiting": 0,
        "kv_cache_active_usage_perc": 0.0,
        "kv_cache_cached_free_usage_perc": 25.0,
        "kv_cache_resident_usage_perc": 25.0,
        "cached_block_count": 4,
        "free_cached_block_count": 2,
        "free_uncached_block_count": 6,
    }

    with patch.object(metric_loggers.logger, "info") as info_mock:
        logging_stat_logger.log()

    assert info_mock.call_count == 1
    args = info_mock.call_args[0]
    assert "GPU KV cache usage: active %.1f%%, cached free %.1f%%, resident %.1f%%" in args[0]
    assert "Cached blocks: %d, Free cached blocks: %d, Cold free blocks: %d" in args[0]
    assert args[1] == 2
    assert args[4] == 0
    assert args[5] == 0
    assert args[6] == 0.0
    assert args[7] == 25.0
    assert args[8] == 25.0
    assert args[9] == 4
    assert args[10] == 2
    assert args[11] == 6


def test_logging_stat_logger_records_throughput_trace_samples():
    import vllm.v1.metrics.loggers as metric_loggers
    from vllm.v1.metrics.stats import SchedulerStats

    patch_vllm_engine_core_inspection()

    logging_stat_logger = metric_loggers.LoggingStatLogger(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks=8)),
        engine_index=2,
    )
    logging_stat_logger.num_prompt_tokens = 100
    logging_stat_logger.num_generation_tokens = 50
    logging_stat_logger._get_throughput = lambda token_count, now: 12.5 if token_count == 100 else 3.5
    logging_stat_logger._reset = lambda now: None
    logging_stat_logger.last_prompt_throughput = 0.0
    logging_stat_logger.last_generation_throughput = 0.0
    logging_stat_logger.last_scheduler_stats = SchedulerStats(
        num_running_reqs=1,
        num_waiting_reqs=0,
        kv_cache_usage=0.25,
    )
    logging_stat_logger._roll_active_trace_steps = (11,)
    logging_stat_logger._roll_trace_component = "infer_worker_metrics"

    class _FakeTracer:
        def __init__(self):
            self.samples = []

        def record_sample(self, *args, **kwargs):
            self.samples.append((args, kwargs))

    fake_tracer = _FakeTracer()

    with patch("roll.third_party.vllm.engine_core_patch.get_trace_manager", return_value=fake_tracer), patch.object(
        metric_loggers.logger, "info"
    ):
        logging_stat_logger.log()

    assert len(fake_tracer.samples) == 2
    sample_values = {args[0]: args[1] for args, _ in fake_tracer.samples}
    assert sample_values["vllm.prompt_throughput_tps"] == 12.5
    assert sample_values["vllm.generation_throughput_tps"] == 3.5
    for _, kwargs in fake_tracer.samples:
        assert kwargs["step"] == 11
        assert kwargs["attrs"]["source"] == "logger_manager"
        assert kwargs["attrs"]["engine"] == "2"


def test_prometheus_stat_logger_records_kv_usage_breakdown_gauges():
    import vllm.v1.metrics.loggers as metric_loggers
    from vllm.v1.metrics.stats import SchedulerStats

    patch_vllm_engine_core_inspection()

    class _FakePromChild:
        def __init__(self):
            self.set_calls = []
            self.inc_calls = []
            self.observe_calls = []

        def set(self, value):
            self.set_calls.append(value)

        def inc(self, value=1):
            self.inc_calls.append(value)

        def observe(self, value):
            self.observe_calls.append(value)

        def set_to_current_time(self):
            return None

    class _FakePromMetric:
        def __init__(self, *args, **kwargs):
            self.children = {}

        def labels(self, *args, **kwargs):
            key = args if args else tuple(sorted(kwargs.items()))
            self.children.setdefault(key, _FakePromChild())
            return self.children[key]

    class _FakeSpecDecodingProm:
        def __init__(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

    fake_config = SimpleNamespace(
        observability_config=SimpleNamespace(show_hidden_metrics=False),
        model_config=SimpleNamespace(served_model_name="unit-test-model", max_model_len=128),
        speculative_config=None,
        lora_config=None,
    )

    with patch.object(metric_loggers.PrometheusStatLogger, "_gauge_cls", _FakePromMetric), patch.object(
        metric_loggers.PrometheusStatLogger, "_counter_cls", _FakePromMetric
    ), patch.object(metric_loggers.PrometheusStatLogger, "_histogram_cls", _FakePromMetric), patch.object(
        metric_loggers.PrometheusStatLogger, "_spec_decoding_cls", _FakeSpecDecodingProm
    ), patch.object(metric_loggers, "unregister_vllm_metrics"):
        prometheus_logger = metric_loggers.PrometheusStatLogger(fake_config, [0])

    scheduler_stats = SchedulerStats(
        num_running_reqs=2,
        num_waiting_reqs=1,
        kv_cache_usage=0.5,
    )
    scheduler_stats.kv_cache_cached_free_usage = 0.25
    scheduler_stats.kv_cache_resident_usage = 0.75

    prometheus_logger.record(scheduler_stats=scheduler_stats, iteration_stats=None, engine_idx=0)

    assert prometheus_logger.gauge_kv_cache_cached_free_usage[0].set_calls == [0.25]
    assert prometheus_logger.gauge_kv_cache_resident_usage[0].set_calls == [0.75]
