import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from roll.distributed.strategy.vllm_strategy import VllmStrategy


class _FakeTracer:
    def record_sample(self, *args, **kwargs):
        return None


def _build_worker():
    worker = Mock()
    worker.worker_config = Mock()
    worker.worker_config.offload_nccl = False
    worker.pipeline_config = Mock()
    worker.pipeline_config.seed = 0
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
