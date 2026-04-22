import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from roll.distributed.strategy.vllm_strategy import VllmStrategy


class _FakeTracer:
    def __init__(self):
        self.samples = []

    def record_sample(self, *args, **kwargs):
        self.samples.append((args, kwargs))
        return None


def _build_worker():
    worker = Mock()
    worker.worker_config = Mock()
    worker.worker_config.offload_nccl = False
    worker.pipeline_config = Mock()
    worker.pipeline_config.seed = 0
    worker.pipeline_config.is_actor_infer_colocated = True
    worker.worker_name = "infer_worker"
    worker.cluster_name = "actor_infer"
    worker.rank = 0
    worker.world_size = 1
    return worker


@pytest.mark.anyio
async def test_collect_metrics_snapshot_skips_empty_realtime_keys_without_kv_snapshot():
    strategy = VllmStrategy(_build_worker())
    strategy._collect_engine_core_kv_snapshots = AsyncMock(return_value=[])

    metric = SimpleNamespace(name="vllm:num_requests_waiting", value=3.0, labels={})

    with patch(
        "vllm.v1.metrics.reader.get_metrics_snapshot",
        return_value=[metric],
    ), patch(
        "roll.distributed.strategy.vllm_strategy.get_trace_manager",
        return_value=_FakeTracer(),
    ), patch(
        "roll.distributed.strategy.vllm_strategy.asyncio.sleep",
        side_effect=asyncio.CancelledError,
    ):
        with pytest.raises(asyncio.CancelledError):
            await strategy._collect_metrics_snapshot()

    assert len(strategy._metrics_snapshots) == 1
    snapshot = strategy._metrics_snapshots[0]
    assert snapshot == {"vllm/num_requests_waiting_max": [3.0]}
    assert "vllm/kv_cache_usage_perc_max" not in snapshot
    assert strategy.get_metrics() == {"vllm/num_requests_waiting_max": 3.0}


@pytest.mark.anyio
async def test_weight_updates_reset_engine_core_lmcache_once_per_model_update():
    strategy = VllmStrategy(_build_worker())
    strategy._lmcache_enabled = True
    strategy.model = Mock()
    strategy.model.call_engine_core_utility = AsyncMock()
    strategy.model.broadcast_parameter = AsyncMock()
    strategy.model.update_parameter_in_bucket = AsyncMock()
    strategy.model.process_weights_after_loading = AsyncMock()
    strategy.model.reset_prefix_cache = AsyncMock()
    strategy.model.offload_states = AsyncMock()
    strategy.is_model_in_gpu = True
    strategy.sleep_level = 1

    with patch("roll.distributed.strategy.vllm_strategy.current_platform.empty_cache", return_value=None):
        await strategy.broadcast_parameter(["w"], ["float32"], [(1,)], "g", False)
        await strategy.update_parameter_in_bucket([b"payload"], False)

        assert strategy.model.call_engine_core_utility.await_count == 1
        strategy.model.call_engine_core_utility.assert_awaited_with("roll_reset_lmcache_engine", "weight update")

        await strategy.process_weights_after_loading()
        assert strategy.model.call_engine_core_utility.await_count == 1
        assert strategy._engine_core_lmcache_invalidated is False

        await strategy.offload_states()
        assert strategy.model.call_engine_core_utility.await_count == 2
        strategy.model.call_engine_core_utility.assert_any_await("roll_reset_lmcache_engine", "offload_states")


@pytest.mark.anyio
async def test_collect_metrics_snapshot_records_kv_usage_breakdown_from_engine_snapshot():
    strategy = VllmStrategy(_build_worker())
    strategy._active_trace_steps = {17: object()}
    strategy._collect_engine_core_kv_snapshots = AsyncMock(
        return_value=[
            {
                "engine": "0",
                "kv_cache_cached_free_usage_perc": 12.5,
                "kv_cache_resident_usage_perc": 62.5,
            }
        ]
    )
    fake_tracer = _FakeTracer()

    with patch(
        "vllm.v1.metrics.reader.get_metrics_snapshot",
        return_value=[],
    ), patch(
        "roll.distributed.strategy.vllm_strategy.get_trace_manager",
        return_value=fake_tracer,
    ), patch(
        "roll.distributed.strategy.vllm_strategy.asyncio.sleep",
        side_effect=asyncio.CancelledError,
    ):
        with pytest.raises(asyncio.CancelledError):
            await strategy._collect_metrics_snapshot()

    assert len(strategy._metrics_snapshots) == 1
    snapshot = strategy._metrics_snapshots[0]
    assert snapshot["vllm/kv_cache_cached_free_usage_perc_max"] == [12.5]
    assert snapshot["vllm/kv_cache_resident_usage_perc_max"] == [62.5]
    recorded_names = {args[0] for args, _ in fake_tracer.samples}
    assert "vllm.kv_cache_cached_free_usage_pct" in recorded_names
    assert "vllm.kv_cache_resident_usage_pct" in recorded_names


@pytest.mark.anyio
async def test_collect_metrics_snapshot_records_prompt_token_rate_sample():
    strategy = VllmStrategy(_build_worker())
    strategy._active_trace_steps = {23: object()}
    strategy._collect_engine_core_kv_snapshots = AsyncMock(return_value=[])
    strategy.model = SimpleNamespace(logger_manager=SimpleNamespace(per_engine_logger_dict={}))
    strategy._counter_last_samples = {
        ("vllm:prompt_tokens_total", ()): (1_000_000_000, 100.0),
    }
    fake_tracer = _FakeTracer()
    metric = SimpleNamespace(name="vllm:prompt_tokens_total", value=160.0, labels={})

    with patch(
        "vllm.v1.metrics.reader.get_metrics_snapshot",
        return_value=[metric],
    ), patch(
        "roll.distributed.strategy.vllm_strategy.get_trace_manager",
        return_value=fake_tracer,
    ), patch(
        "roll.distributed.strategy.vllm_strategy.time.time_ns",
        return_value=2_000_000_000,
    ), patch(
        "roll.distributed.strategy.vllm_strategy.asyncio.sleep",
        side_effect=asyncio.CancelledError,
    ):
        with pytest.raises(asyncio.CancelledError):
            await strategy._collect_metrics_snapshot()

    assert len(strategy._metrics_snapshots) == 1
    snapshot = strategy._metrics_snapshots[0]
    assert snapshot["vllm/prompt_tokens_rate_tps"] == 60.0

    sample_values = {args[0]: args[1] for args, _ in fake_tracer.samples}
    assert sample_values["vllm.prompt_tokens_rate_tps"] == 60.0


def test_activate_trace_step_syncs_logger_trace_state():
    strategy = VllmStrategy(_build_worker())
    logger_a = SimpleNamespace()
    logger_b = SimpleNamespace()
    strategy.model = SimpleNamespace(
        logger_manager=SimpleNamespace(
            per_engine_logger_dict={
                0: [logger_a],
                1: [logger_b],
            }
        )
    )

    strategy._activate_trace_step(7)

    assert logger_a._roll_active_trace_steps == (7,)
    assert logger_b._roll_active_trace_steps == (7,)
    assert logger_a._roll_trace_component == "infer_worker_metrics"
    assert logger_b._roll_trace_component == "infer_worker_metrics"

    strategy._activate_trace_step(9)

    assert logger_a._roll_active_trace_steps == (7, 9)
    assert logger_b._roll_active_trace_steps == (7, 9)

    strategy._deactivate_trace_step(7)

    assert logger_a._roll_active_trace_steps == (9,)
    assert logger_b._roll_active_trace_steps == (9,)

    strategy._deactivate_trace_step(9)

    assert logger_a._roll_active_trace_steps == ()
    assert logger_b._roll_active_trace_steps == ()
