from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.router import RouterClient
from roll.distributed.strategy.vllm_strategy import VllmStrategy


def _build_worker():
    worker = Mock()
    worker.worker_config = Mock()
    worker.worker_config.offload_nccl = False
    worker.worker_config.model_args.model_name_or_path = "model@sha256"
    worker.worker_config.model_args.lora_target = None
    worker.pipeline_config = SimpleNamespace(
        seed=0,
        is_actor_infer_colocated=True,
        agentic_kv=SimpleNamespace(
            enable=True,
            free_gpu_blocks_low_watermark=8,
            free_gpu_blocks_high_watermark=16,
            max_cached_free_blocks=0,
        ),
    )
    worker.worker_name = "infer_worker"
    worker.cluster_name = "actor_infer"
    worker.rank = 0
    worker.world_size = 1
    return worker


def test_router_client_forwards_agentic_kv_blob():
    client = RouterClient(
        proxy=None,
        meta={
            "strategy_name": "vllm",
            "eos_token_id": 7,
            "pad_token_id": 0,
        },
    )
    req = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            },
            batch_size=[1],
        ),
        meta_info={
            "generation_config": {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 50,
                "num_return_sequences": 1,
                "repetition_penalty": 1.0,
                "stop_strings": ["</tool_call>"],
            },
            "agentic_kv": {
                "session_id": "traj-1",
                "candidate_resume_point_id": "traj-1:rp:2",
                "request_configs": {"lmcache.tag.akv_session": "traj-1"},
            },
        },
    )

    payload, request_id = client._preprocess_generate(req, request_id="req-2")

    assert request_id == "req-2"
    assert payload["agentic_kv"]["session_id"] == "traj-1"
    assert payload["agentic_kv"]["candidate_resume_point_id"] == "traj-1:rp:2"


def test_router_client_postprocess_forwards_agentic_kv_blob():
    client = RouterClient(
        proxy=None,
        meta={
            "strategy_name": "vllm",
            "eos_token_id": 7,
            "pad_token_id": 0,
        },
    )
    req = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            },
            batch_size=[1],
        ),
        meta_info={"generation_config": {"max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0, "top_k": 50, "num_return_sequences": 1, "repetition_penalty": 1.0, "stop_strings": []}},
    )

    output = client._postprocess_generate(
        req,
        response={
            "finish_reasons": ["stop"],
            "output_token_ids": [[4, 5]],
            "output_logprobs": [],
            "agentic_kv": {
                "session_id": "traj-1",
                "save_on_wait": True,
                "evicted_cached_free_blocks": 8,
            },
        },
        request_id="req-2",
    )

    assert output.meta_info["agentic_kv"] == {
        "session_id": "traj-1",
        "save_on_wait": True,
        "evicted_cached_free_blocks": 8,
    }


def test_merge_kv_transfer_params_appends_agentic_kv_request_tags():
    strategy = VllmStrategy.__new__(VllmStrategy)

    merged = strategy._merge_kv_transfer_params(
        {"max_tokens": 16},
        {
            "lmcache.tag.akv_session": "traj-1",
            "lmcache.skip_save": False,
        },
    )

    assert merged["extra_args"]["kv_transfer_params"]["lmcache.tag.akv_session"] == "traj-1"
    assert merged["extra_args"]["kv_transfer_params"]["lmcache.skip_save"] is False


@pytest.mark.anyio
async def test_should_save_waiting_session_forces_save_in_active_only_hbm_mode():
    strategy = VllmStrategy(_build_worker())
    strategy._collect_engine_core_kv_snapshots = AsyncMock(return_value=[])

    assert await strategy._should_save_waiting_session() is True


@pytest.mark.anyio
async def test_generate_request_forces_save_and_evicts_cached_free_blocks():
    strategy = VllmStrategy(_build_worker())
    strategy.is_lora = False

    async def _generate(*args, **kwargs):
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[11, 12], finish_reason="stop", logprobs=None)],
            roll_phase_timing=None,
        )

    strategy.model = SimpleNamespace(
        generate=_generate,
        call_engine_core_utility=AsyncMock(
            return_value={
                "available": True,
                "evicted_blocks": 8,
                "free_cached_before": 8,
                "free_cached_after": 0,
                "target_free_cached_blocks": 0,
            }
        ),
    )
    strategy._activate_trace_step = lambda step: step
    strategy._deactivate_trace_step = lambda step: None

    with patch("roll.distributed.strategy.vllm_strategy.SamplingParams", side_effect=lambda **kwargs: kwargs), patch(
        "roll.distributed.strategy.vllm_strategy.logger.info"
    ) as info_mock:
        response = await strategy.generate_request(
            {
                "rid": "req-1",
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_tokens": 4, "temperature": 0.0, "top_p": 1.0, "top_k": 50},
                "agentic_kv": {
                    "session_id": "traj-1",
                    "candidate_resume_point_id": "traj-1:rp:1",
                    "boundary_kind": "request_end_tool_call",
                    "wait_reason": "tool_wait",
                    "request_configs": {"lmcache.tag.akv_session": "traj-1"},
                },
            }
    )

    assert response["agentic_kv"]["save_on_wait"] is True
    assert response["agentic_kv"]["evicted_cached_free_blocks"] == 8
    strategy.model.call_engine_core_utility.assert_awaited_with("roll_evict_cached_free_blocks", 0)
    assert info_mock.call_count == 1
    assert "AgenticKV active-only eviction committed" in info_mock.call_args[0][0]


@pytest.mark.anyio
async def test_generate_request_does_not_log_when_active_only_evicts_nothing():
    strategy = VllmStrategy(_build_worker())
    strategy.is_lora = False

    async def _generate(*args, **kwargs):
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[11, 12], finish_reason="stop", logprobs=None)],
            roll_phase_timing=None,
        )

    strategy.model = SimpleNamespace(
        generate=_generate,
        call_engine_core_utility=AsyncMock(
            return_value={
                "available": True,
                "evicted_blocks": 0,
                "free_cached_before": 0,
                "free_cached_after": 0,
                "target_free_cached_blocks": 0,
            }
        ),
    )
    strategy._activate_trace_step = lambda step: step
    strategy._deactivate_trace_step = lambda step: None

    with patch("roll.distributed.strategy.vllm_strategy.SamplingParams", side_effect=lambda **kwargs: kwargs), patch(
        "roll.distributed.strategy.vllm_strategy.logger.info"
    ) as info_mock:
        response = await strategy.generate_request(
            {
                "rid": "req-1",
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_tokens": 4, "temperature": 0.0, "top_p": 1.0, "top_k": 50},
                "agentic_kv": {
                    "session_id": "traj-1",
                    "candidate_resume_point_id": "traj-1:rp:1",
                    "boundary_kind": "request_end_tool_call",
                    "wait_reason": "tool_wait",
                    "request_configs": {"lmcache.tag.akv_session": "traj-1"},
                },
            }
        )

    assert response["agentic_kv"]["save_on_wait"] is True
    assert response["agentic_kv"]["evicted_cached_free_blocks"] == 0
    assert info_mock.call_count == 0
