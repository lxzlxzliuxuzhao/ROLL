import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import zmq
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.v1.config import load_engine_config_with_overrides
from roll.third_party.vllm import _configure_lmcache
from roll.third_party.vllm.worker import WorkerBase
from lmcache.integration.vllm.utils import calculate_local_rank_and_world_size
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
import torch


def test_configure_lmcache_sets_env_and_transfer_config():
    kwargs = {
        "model": "dummy-model",
        "lmcache_config": {
            "chunk_size": 256,
            "local_cpu": True,
            "max_local_cpu_size": 16.0,
            "kv_connector_extra_config": {"discard_partial_chunks": False},
        },
    }

    with patch.dict(os.environ, {}, clear=False), patch(
        "roll.third_party.vllm.uuid.uuid4",
        return_value=SimpleNamespace(hex="abc123"),
    ):
        enabled = _configure_lmcache(kwargs)

        assert enabled is True
        assert "lmcache_config" not in kwargs
        assert os.environ["LMCACHE_CHUNK_SIZE"] == "256"
        assert os.environ["LMCACHE_LOCAL_CPU"] == "True"
        assert os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "16.0"

        kv_transfer_config = kwargs["kv_transfer_config"]
        assert kv_transfer_config.kv_connector == "LMCacheConnectorV1Dynamic"
        assert kv_transfer_config.kv_role == "kv_both"
        assert kv_transfer_config.engine_id == "roll-lmcache-abc123"
        assert (
            kv_transfer_config.kv_connector_module_path
            == "lmcache.integration.vllm.lmcache_connector_v1"
        )
        assert kv_transfer_config.kv_connector_extra_config == {
            "discard_partial_chunks": False
        }


def test_configure_lmcache_rejects_duplicate_transfer_config():
    kwargs = {
        "lmcache_config": {},
        "kv_transfer_config": object(),
    }

    with patch.dict(os.environ, {}, clear=False):
        try:
            _configure_lmcache(kwargs)
        except ValueError as e:
            assert "lmcache_config" in str(e)
        else:
            raise AssertionError("expected ValueError for duplicated kv config")


def test_configure_lmcache_serializes_complex_env_values():
    kwargs = {
        "model": "dummy-model",
        "lmcache_config": {
            "chunk_size": 256,
            "local_cpu": True,
            "max_local_cpu_size": 16.0,
            "lookup_server_worker_ids": [0, 3],
            "extra_config": {"foo": "bar"},
        },
    }

    with patch.dict(os.environ, {}, clear=False):
        _configure_lmcache(kwargs)

        assert os.environ["LMCACHE_LOOKUP_SERVER_WORKER_IDS"] == "0,3"
        assert json.loads(os.environ["LMCACHE_EXTRA_CONFIG"]) == {"foo": "bar"}


def test_worker_offload_states_resets_lmcache_connector():
    worker = WorkerBase()
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector="LMCacheConnectorV1")
    )
    worker.sleep = MagicMock()
    worker.custom_init_worker()
    connector = MagicMock()

    with patch(
        "roll.third_party.vllm.worker.current_platform.empty_cache"
    ), patch(
        "vllm.distributed.kv_transfer.has_kv_transfer_group", return_value=True
    ), patch(
        "vllm.distributed.kv_transfer.get_kv_transfer_group", return_value=connector
    ):
        worker.offload_states(level=1)

    connector.reset_lmcache_engine.assert_called_once()
    worker.sleep.assert_called_once_with(1)
    assert worker.weight_loaded is False
    assert worker.kv_cache_loaded is False


def test_worker_load_weights_resets_lmcache_connector_once_per_model_update():
    worker = WorkerBase()
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector="LMCacheConnectorV1")
    )
    worker.model_runner = SimpleNamespace(model=MagicMock())
    worker.custom_init_worker()
    connector = MagicMock()

    with patch(
        "vllm.distributed.kv_transfer.has_kv_transfer_group", return_value=True
    ), patch(
        "vllm.distributed.kv_transfer.get_kv_transfer_group", return_value=connector
    ):
        worker.load_weights([("weight", torch.zeros(1))])
        worker.load_weights([("weight", torch.ones(1))])

        connector.reset_lmcache_engine.assert_called_once()

        worker.process_weights_after_loading()
        worker.load_weights([("weight", torch.full((1,), 2.0))])

    assert connector.reset_lmcache_engine.call_count == 2
    assert worker.model_runner.model.load_weights.call_count == 3


def test_worker_lora_weight_updates_reset_lmcache_connector_on_each_model_update():
    worker = WorkerBase()
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector="LMCacheConnectorV1")
    )
    worker.device = "cpu"
    worker.rank = 0
    worker.custom_init_worker()
    connector = MagicMock()

    class _Handle:
        def wait(self):
            return None

    with patch(
        "roll.third_party.vllm.worker.collective.broadcast",
        return_value=_Handle(),
    ), patch(
        "roll.third_party.vllm.worker.MultiprocessingSerializer.deserialize",
        return_value={"bucket": None, "tensors_meta": None},
    ), patch(
        "roll.third_party.vllm.worker.named_tensors_from_bucket",
        return_value=[("lora-b", torch.ones(1))],
    ), patch(
        "roll.third_party.vllm.worker.monkey_patch_torch_reductions"
    ), patch(
        "roll.third_party.vllm.worker.vllm.__version__", "0.12.0"
    ), patch(
        "vllm.distributed.kv_transfer.has_kv_transfer_group", return_value=True
    ), patch(
        "vllm.distributed.kv_transfer.get_kv_transfer_group", return_value=connector
    ):
        worker.broadcast_parameter(
            names=["lora-a"],
            dtypes=["float32"],
            shapes=[(1,)],
            group_name="test-group",
            is_lora=True,
        )
        assert worker.tensor_lora_manager.lora_params["lora-a"].shape == torch.Size([1])
        connector.reset_lmcache_engine.assert_called_once()

        worker.process_weights_after_loading()

        worker.update_parameter_in_bucket([b"payload"], is_lora=True)
        assert worker.tensor_lora_manager.lora_params["lora-b"].shape == torch.Size([1])

    assert connector.reset_lmcache_engine.call_count == 2


def test_zmq_offload_server_close_stops_thread_cleanly():
    engine = SimpleNamespace(
        metadata=SimpleNamespace(engine_id="unit-test-engine"),
        store=MagicMock(),
    )

    class _FakeSocket:
        def __init__(self):
            self.closed = False
            self.recv_timeout_ms = 0

        def setsockopt(self, option, value):
            if option == zmq.RCVTIMEO:
                self.recv_timeout_ms = value

        def recv(self, copy=False):
            time.sleep(max(self.recv_timeout_ms, 1) / 1000.0)
            raise zmq.Again()

        def send(self, _response):
            return None

        def close(self, linger=0):
            self.closed = True

    fake_socket = _FakeSocket()

    with patch(
        "lmcache.v1.offload_server.zmq_server.get_zmq_context",
        return_value=MagicMock(),
    ), patch(
        "lmcache.v1.offload_server.zmq_server.get_zmq_socket",
        return_value=fake_socket,
    ):
        server = ZMQOffloadServer(engine, tp_rank=0)
        server.close()

    assert server.thread.is_alive() is False
    assert server._socket_closed is True
    assert fake_socket.closed is True


def test_calculate_local_rank_and_world_size_single_visible_device():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=1,
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        )
    )

    class TorchDev:
        @staticmethod
        def device_count():
            return 1

    with patch(
        "lmcache.integration.vllm.utils.get_vllm_torch_dev",
        return_value=(TorchDev(), "cuda"),
    ):
        local_rank, local_world_size = calculate_local_rank_and_world_size(vllm_config)

    assert local_rank == 0
    assert local_world_size == 1


class _DummyKVTransferConfig(SimpleNamespace):
    def get_from_extra_config(self, _key, default=None):
        return default


class _DummyParentConnector:
    def register_kv_caches(self, kv_caches):
        return kv_caches


def _build_minimal_vllm_config():
    return SimpleNamespace(
        device_config=SimpleNamespace(device="cuda"),
        kv_transfer_config=_DummyKVTransferConfig(
            kv_role="kv_both",
            kv_connector_extra_config={},
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
        ),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(
            get_num_layers=lambda _parallel_config: 32,
        ),
    )


def test_worker_connector_eagerly_post_inits_lmcache():
    config = load_engine_config_with_overrides(
        overrides={
            "chunk_size": 256,
            "local_cpu": True,
            "max_local_cpu_size": 16.0,
            "enable_async_loading": False,
            "use_layerwise": False,
        }
    )
    manager = MagicMock()
    manager.lmcache_engine = None
    manager.lookup_client = None
    manager.lookup_server = None
    manager.lmcache_engine_metadata = None

    with patch(
        "lmcache.integration.vllm.vllm_v1_adapter.lmcache_get_or_create_config",
        return_value=config,
    ), patch(
        "lmcache.integration.vllm.vllm_v1_adapter.LMCacheManager",
        return_value=manager,
    ):
        LMCacheConnectorV1Impl(
            _build_minimal_vllm_config(),
            KVConnectorRole.WORKER,
            _DummyParentConnector(),
        )

    manager.start_services.assert_called_once()
    manager.post_init.assert_called_once()


def test_scheduler_connector_does_not_eagerly_post_init_lmcache():
    config = load_engine_config_with_overrides(
        overrides={
            "chunk_size": 256,
            "local_cpu": True,
            "max_local_cpu_size": 16.0,
            "enable_async_loading": False,
            "use_layerwise": False,
        }
    )
    manager = MagicMock()
    manager.lmcache_engine = None
    manager.lookup_client = None
    manager.lookup_server = None
    manager.lmcache_engine_metadata = None

    with patch(
        "lmcache.integration.vllm.vllm_v1_adapter.lmcache_get_or_create_config",
        return_value=config,
    ), patch(
        "lmcache.integration.vllm.vllm_v1_adapter.LMCacheManager",
        return_value=manager,
    ):
        LMCacheConnectorV1Impl(
            _build_minimal_vllm_config(),
            KVConnectorRole.SCHEDULER,
            _DummyParentConnector(),
        )

    manager.start_services.assert_called_once()
    manager.post_init.assert_not_called()


def test_worker_connector_reset_recreates_manager_and_reregisters_kv_caches():
    config = load_engine_config_with_overrides(
        overrides={
            "chunk_size": 256,
            "local_cpu": True,
            "max_local_cpu_size": 16.0,
            "enable_async_loading": False,
            "use_layerwise": False,
        }
    )
    manager1 = MagicMock()
    manager1.lmcache_engine = None
    manager1.lookup_client = None
    manager1.lookup_server = None
    manager1.lmcache_engine_metadata = None
    manager2 = MagicMock()
    manager2.lmcache_engine = None
    manager2.lookup_client = None
    manager2.lookup_server = None
    manager2.lmcache_engine_metadata = None

    with patch(
        "lmcache.integration.vllm.vllm_v1_adapter.lmcache_get_or_create_config",
        return_value=config,
    ), patch(
        "lmcache.integration.vllm.vllm_v1_adapter.LMCacheManager",
        side_effect=[manager1, manager2],
    ):
        connector = LMCacheConnectorV1Impl(
            _build_minimal_vllm_config(),
            KVConnectorRole.WORKER,
            _DummyParentConnector(),
        )
        connector._build_kv_layer_groups = MagicMock()
        connector.kv_caches = {"model.layers.0.attn": torch.zeros(1)}
        manager1.post_init.reset_mock()
        connector.reset("unit-test")

    manager1.stop_services.assert_called_once()
    manager2.start_services.assert_called_once()
    manager2.post_init.assert_called_once()
    assert "model.layers.0.attn" in connector.kv_caches
