import asyncio
import copy
import gc
import os
import time
from collections import deque
from typing import Dict, List, Optional
from packaging.version import Version

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
import vllm
from vllm import RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, BeamSearchParams
from vllm.inputs.data import TokensPrompt
from vllm.utils import random_uuid

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto, list_of_dict_to_dict_of_list
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.third_party.vllm import create_async_llm
from roll.utils.functionals import (
    concatenate_input_and_output,
    reduce_metrics,
    gather_unpadded_input_ids,
)
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType
from roll.platforms import current_platform
from roll.utils.tracing.core import get_trace_manager


logger = get_logger()


_VLLM_GAUGE_METRIC_SPECS = {
    "vllm:num_requests_waiting": {
        "snapshot_key": "vllm/num_requests_waiting_max",
        "sample_name": "vllm.num_requests_waiting",
        "unit": "req",
    },
    "vllm:num_requests_running": {
        "snapshot_key": "vllm/num_requests_running_max",
        "sample_name": "vllm.num_requests_running",
        "unit": "req",
    },
}

_VLLM_COUNTER_METRIC_SPECS = {
    "vllm:num_preemptions": {
        "snapshot_key": "vllm/num_preemptions_delta@sum",
        "sample_name": "vllm.num_preemptions_delta",
        "unit": "count",
    },
    "vllm:prefix_cache_queries": {
        "snapshot_key": "vllm/prefix_cache_queries_delta@sum",
        "sample_name": "vllm.prefix_cache_queries_delta",
        "unit": "tok",
    },
    "vllm:prefix_cache_hits": {
        "snapshot_key": "vllm/prefix_cache_hits_delta@sum",
        "sample_name": "vllm.prefix_cache_hits_delta",
        "unit": "tok",
    },
}

_VLLM_COUNTER_RATE_METRIC_SPECS = {
    "vllm:prompt_tokens_total": {
        "snapshot_key": "vllm/prompt_tokens_rate_tps",
        "sample_name": "vllm.prompt_tokens_rate_tps",
        "unit": "tok/s",
    },
}

_VLLM_REALTIME_SAMPLE_SPECS = {
    "prompt_throughput_tps": {
        "snapshot_key": "vllm/prompt_throughput_tps",
        "sample_name": "vllm.prompt_throughput_tps",
        "unit": "tok/s",
    },
    "generation_throughput_tps": {
        "snapshot_key": "vllm/generation_throughput_tps",
        "sample_name": "vllm.generation_throughput_tps",
        "unit": "tok/s",
    },
    "prompt_tokens_rate_tps": {
        "snapshot_key": "vllm/prompt_tokens_rate_tps",
        "sample_name": "vllm.prompt_tokens_rate_tps",
        "unit": "tok/s",
    },
    "kv_cache_usage_perc": {
        "snapshot_key": "vllm/kv_cache_usage_perc_max",
        "sample_name": "vllm.kv_cache_usage_pct",
        "unit": "%",
    },
    "kv_cache_cached_free_usage_perc": {
        "snapshot_key": "vllm/kv_cache_cached_free_usage_perc_max",
        "sample_name": "vllm.kv_cache_cached_free_usage_pct",
        "unit": "%",
    },
    "kv_cache_resident_usage_perc": {
        "snapshot_key": "vllm/kv_cache_resident_usage_perc_max",
        "sample_name": "vllm.kv_cache_resident_usage_pct",
        "unit": "%",
    },
    "num_gpu_blocks_total": {
        "snapshot_key": "vllm/kv_blocks_total_max",
        "sample_name": "vllm.kv_blocks_total",
        "unit": "blocks",
    },
    "num_gpu_blocks_used": {
        "snapshot_key": "vllm/kv_blocks_used_max",
        "sample_name": "vllm.kv_blocks_used",
        "unit": "blocks",
    },
    "num_gpu_blocks_free": {
        "snapshot_key": "vllm/kv_blocks_free_min",
        "sample_name": "vllm.kv_blocks_free",
        "unit": "blocks",
    },
    "cached_block_entries": {
        "snapshot_key": "vllm/kv_cached_entries_max",
        "sample_name": "vllm.kv_cached_entries",
        "unit": "entries",
    },
    "cached_block_count": {
        "snapshot_key": "vllm/kv_cached_blocks_max",
        "sample_name": "vllm.kv_cached_blocks",
        "unit": "blocks",
    },
    "free_cached_block_count": {
        "snapshot_key": "vllm/kv_blocks_free_cached_max",
        "sample_name": "vllm.kv_blocks_free_cached",
        "unit": "blocks",
    },
    "free_uncached_block_count": {
        "snapshot_key": "vllm/kv_blocks_free_uncached_min",
        "sample_name": "vllm.kv_blocks_free_uncached",
        "unit": "blocks",
    },
    "kv_cache_total_bytes": {
        "snapshot_key": "vllm/kv_cache_total_bytes_max",
        "sample_name": "vllm.kv_cache_total_bytes",
        "unit": "bytes",
    },
    "kv_cache_allocated_bytes": {
        "snapshot_key": "vllm/kv_cache_allocated_bytes_max",
        "sample_name": "vllm.kv_cache_allocated_bytes",
        "unit": "bytes",
    },
    "kv_cache_used_bytes": {
        "snapshot_key": "vllm/kv_cache_used_bytes_max",
        "sample_name": "vllm.kv_cache_used_bytes",
        "unit": "bytes",
    },
    "kv_cache_free_bytes": {
        "snapshot_key": "vllm/kv_cache_free_bytes_min",
        "sample_name": "vllm.kv_cache_free_bytes",
        "unit": "bytes",
    },
    "kv_cache_reserved_bytes": {
        "snapshot_key": "vllm/kv_cache_reserved_bytes_max",
        "sample_name": "vllm.kv_cache_reserved_bytes",
        "unit": "bytes",
    },
    "num_requests_running": {
        "snapshot_key": "vllm/num_requests_running_max",
        "sample_name": "vllm.num_requests_running",
        "unit": "req",
    },
    "num_requests_waiting": {
        "snapshot_key": "vllm/num_requests_waiting_max",
        "sample_name": "vllm.num_requests_waiting",
        "unit": "req",
    },
    "num_preemptions_delta": {
        "snapshot_key": "vllm/num_preemptions_delta@sum",
        "sample_name": "vllm.num_preemptions_delta",
        "unit": "count",
    },
    "prefix_cache_queries_delta": {
        "snapshot_key": "vllm/prefix_cache_queries_delta@sum",
        "sample_name": "vllm.prefix_cache_queries_delta",
        "unit": "tok",
    },
    "prefix_cache_hits_delta": {
        "snapshot_key": "vllm/prefix_cache_hits_delta@sum",
        "sample_name": "vllm.prefix_cache_hits_delta",
        "unit": "tok",
    },
    "stored_block_count_delta": {
        "snapshot_key": "vllm/kv_event_stored_blocks@sum",
        "sample_name": "vllm.kv_event_stored_blocks_delta",
        "unit": "blocks",
    },
    "removed_block_count_delta": {
        "snapshot_key": "vllm/kv_event_removed_blocks@sum",
        "sample_name": "vllm.kv_event_removed_blocks_delta",
        "unit": "blocks",
    },
    "cleared_event_count_delta": {
        "snapshot_key": "vllm/kv_event_clears@sum",
        "sample_name": "vllm.kv_event_clears_delta",
        "unit": "count",
    },
}


def _metric_identity(metric) -> tuple[str, tuple[tuple[str, str], ...]]:
    return metric.name, tuple(sorted((metric.labels or {}).items()))


def _metric_sample_attrs(metric) -> dict:
    labels = dict(metric.labels or {})
    return {key: value for key, value in labels.items() if value not in (None, "")}


def _sample_metric_value(metric_name: str, raw_value: float) -> float:
    value = float(raw_value)
    return value


def _lmcache_enabled_from_config(vllm_config: dict) -> bool:
    if "lmcache_config" in vllm_config:
        return True
    kv_transfer_config = vllm_config.get("kv_transfer_config")
    kv_connector = getattr(kv_transfer_config, "kv_connector", None)
    return isinstance(kv_connector, str) and kv_connector.startswith("LMCacheConnectorV1")


class VllmStrategy(InferenceStrategy):
    strategy_name = "vllm"

    def __init__(self, worker: Worker):
        super().__init__(worker)

        # Metrics snapshot infrastructure
        self._metrics_snapshots = deque(maxlen=3600)
        self._metrics_snapshot_interval = 1.0  # Snapshot every 1 second
        self._metrics_task = None
        self._active_trace_steps: dict[int, int] = {}
        self._counter_last_values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._counter_last_samples: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[int, float]] = {}
        self._kv_snapshot_warning_emitted = False
        self._lmcache_enabled = False
        self._lmcache_reset_warning_emitted = False
        self._engine_core_lmcache_invalidated = False
        self._agentic_kv_unload_active = False

    async def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        vllm_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        self._lmcache_enabled = _lmcache_enabled_from_config(vllm_config)
        # Must explicitly set VLLM_USE_V1 to pass this check: https://github.com/vllm-project/vllm/pull/14972
        os.environ["VLLM_USE_V1"] = str(vllm_config.pop("VLLM_USE_V1", 1))
        self.sleep_level = vllm_config.pop("sleep_level", 1)

        data_parallel_size = vllm_config.get("data_parallel_size", 1)
        if data_parallel_size > 1:
            logger.info(
                f"VllmStrategy {self.worker.cluster_name} enable data parallel {data_parallel_size=} data_parallel_rank={self.worker.rank}"
                f" data_parallel_address={os.environ['MASTER_ADDR']} data_parallel_rpc_port={os.environ['MASTER_PORT']}"
            )
            assert data_parallel_size == self.worker.world_size, f"{data_parallel_size=} != {self.worker.world_size=}"
            vllm_config.update(
                {
                    "data_parallel_rank": self.worker.rank, # set data_parallel_rank to use external load balancing
                    "data_parallel_address": os.environ["MASTER_ADDR"],
                    "data_parallel_rpc_port": os.environ["MASTER_PORT"],
                }
            )

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        elif self.worker_config.model_args.dtype == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "auto"
        vllm_config.update(
            {
                "model": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "enforce_eager": vllm_config.get("enforce_eager", False),
                "trust_remote_code": True,
                "seed": self.worker.pipeline_config.seed,
                "disable_custom_all_reduce": vllm_config.get(
                    "disable_custom_all_reduce", True
                ),  # potentially hangs in tp>1
                "enable_prefix_caching": vllm_config.get("enable_prefix_caching", True),
                "load_format": vllm_config.get("load_format", "dummy"),  # use model update passed value
                "max_num_batched_tokens": vllm_config.get("max_num_batched_tokens", 8192), # use default value of LLM class usage context
            }
        )

        self.is_lora = self.worker_config.model_args.lora_target is not None
        if self.is_lora:
            lora_kwargs = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": self.worker_config.model_args.lora_rank,
            }
            vllm_config.update(lora_kwargs)
            vllm_config["load_format"] = "auto"  # enables vLLM to load the base model for add_lora

        logger.info(f"vllm_config: {vllm_config}")
        assert not dist.is_initialized()

        # Can not set VLLM_PORT explicitly in DP. Each call of get_engine_client_zmq_addr in
        # DPCoordinator will return the same port, which will cause port conflict.
        # https://github.com/vllm-project/vllm/blob/releases/v0.10.0/vllm/v1/engine/coordinator.py#L72
        if not data_parallel_size > 1:
            # set VLLM_PORT to avoid port conflict applied by vllm
            vllm_port = self.worker.get_free_port()
            os.environ["VLLM_PORT"] = str(vllm_port)

        self.model = await create_async_llm(resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config)
        self._sync_logger_trace_state()


        if Version("0.15.0") <= Version(vllm.__version__):
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer = await self.model.get_tokenizer()

        assert self.worker.rank_info.dp_rank == self.worker.rank
        assert self.worker.rank_info.dp_size == self.worker.world_size

        self.is_model_in_gpu = True

        try:
            from vllm.v1.metrics.reader import get_metrics_snapshot
            self._metrics_task = asyncio.create_task(self._collect_metrics_snapshot())
        except Exception as e:
            logger.warning(f"Failed to create metrics collector task: {e}")

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        vllm实现compute log probs在这里实现即可
        """
        pass

    @staticmethod
    def _extract_phase_timing(output: Optional[RequestOutput]) -> Optional[Dict]:
        if output is None:
            return None
        phase_timing = getattr(output, "roll_phase_timing", None)
        return dict(phase_timing) if isinstance(phase_timing, dict) else None

    def _merge_kv_transfer_params(self, sampling_params: Dict, kv_transfer_params: Dict) -> Dict:
        merged = dict(sampling_params)
        extra_args = dict(merged.get("extra_args") or {})
        merged_kv_transfer_params = dict(extra_args.get("kv_transfer_params") or {})
        merged_kv_transfer_params.update(kv_transfer_params)
        extra_args["kv_transfer_params"] = merged_kv_transfer_params
        merged["extra_args"] = extra_args
        return merged

    def _agentic_kv_config(self):
        worker = getattr(self, "worker", None)
        pipeline_config = getattr(worker, "pipeline_config", None)
        return getattr(pipeline_config, "agentic_kv", None)

    def _agentic_kv_enabled(self) -> bool:
        agentic_kv = self._agentic_kv_config()
        return bool(agentic_kv and getattr(agentic_kv, "enable", False))

    def _active_only_hbm_target(self) -> Optional[int]:
        agentic_kv = self._agentic_kv_config()
        if agentic_kv is None:
            return None
        target = getattr(agentic_kv, "max_cached_free_blocks", None)
        return None if target is None else int(target)

    def _update_agentic_kv_pressure(self, free_uncached_blocks: int) -> bool:
        agentic_kv = self._agentic_kv_config()
        if agentic_kv is None:
            return False

        low = getattr(agentic_kv, "free_gpu_blocks_low_watermark", None)
        high = getattr(agentic_kv, "free_gpu_blocks_high_watermark", None)
        if low is None or high is None:
            return False

        if self._agentic_kv_unload_active:
            self._agentic_kv_unload_active = free_uncached_blocks < high
        else:
            self._agentic_kv_unload_active = free_uncached_blocks < low
        return self._agentic_kv_unload_active

    def _activate_trace_step(self, step: Optional[int]) -> Optional[int]:
        if step is None:
            return None
        step = int(step)
        self._active_trace_steps[step] = self._active_trace_steps.get(step, 0) + 1
        self._sync_logger_trace_state()
        return step

    def _deactivate_trace_step(self, step: Optional[int]) -> None:
        if step is None:
            return
        remaining = self._active_trace_steps.get(step, 0) - 1
        if remaining > 0:
            self._active_trace_steps[step] = remaining
            self._sync_logger_trace_state()
            return
        self._active_trace_steps.pop(step, None)
        self._sync_logger_trace_state()

    def _sync_logger_trace_state(self) -> None:
        model = getattr(self, "model", None)
        logger_manager = getattr(model, "logger_manager", None)
        per_engine_logger_dict = getattr(logger_manager, "per_engine_logger_dict", {}) or {}
        active_trace_steps = tuple(sorted(int(step) for step in self._active_trace_steps))
        trace_component = f"{self.worker.worker_name}_metrics"
        for stat_loggers in per_engine_logger_dict.values():
            for stat_logger in stat_loggers:
                setattr(stat_logger, "_roll_active_trace_steps", active_trace_steps)
                setattr(stat_logger, "_roll_trace_component", trace_component)

    @staticmethod
    def _normalize_engine_snapshots(raw_snapshot) -> list[dict]:
        if raw_snapshot is None:
            return []
        if isinstance(raw_snapshot, dict):
            return [raw_snapshot]
        if isinstance(raw_snapshot, (list, tuple)):
            return [item for item in raw_snapshot if isinstance(item, dict)]
        return []

    async def _collect_engine_core_kv_snapshots(self) -> list[dict]:
        if not hasattr(self.model, "call_engine_core_utility"):
            return []
        try:
            raw_snapshot = await self.model.call_engine_core_utility("roll_get_kv_cache_snapshot")
        except Exception as exc:
            if not self._kv_snapshot_warning_emitted:
                logger.warning(f"Failed to query real vLLM KV cache snapshot: {exc}")
                self._kv_snapshot_warning_emitted = True
            return []
        snapshots = []
        for engine_index, snapshot in enumerate(self._normalize_engine_snapshots(raw_snapshot)):
            if not snapshot.get("available", False):
                continue
            snapshot = dict(snapshot)
            snapshot["engine"] = str(snapshot.get("engine", engine_index))
            snapshots.append(snapshot)
        return snapshots

    @staticmethod
    def _summarize_agentic_kv_snapshots(snapshots: list[dict]) -> Optional[dict]:
        if not snapshots:
            return None
        total_blocks = sum(int(snapshot.get("num_gpu_blocks_total", 0) or 0) for snapshot in snapshots)
        active_blocks = sum(int(snapshot.get("num_gpu_blocks_used", 0) or 0) for snapshot in snapshots)
        free_cached_blocks = sum(int(snapshot.get("free_cached_block_count", 0) or 0) for snapshot in snapshots)
        free_uncached_blocks = sum(int(snapshot.get("free_uncached_block_count", 0) or 0) for snapshot in snapshots)
        kv_events = [dict(snapshot.get("kv_events") or {}) for snapshot in snapshots]
        stored_block_count_total = sum(int(event.get("stored_block_count_total", 0) or 0) for event in kv_events)
        removed_block_count_total = sum(int(event.get("removed_block_count_total", 0) or 0) for event in kv_events)
        resident_usage_pct = 0.0
        if total_blocks > 0:
            resident_usage_pct = (active_blocks + free_cached_blocks) * 100.0 / total_blocks
        return {
            "total_blocks": total_blocks,
            "active_blocks": active_blocks,
            "free_cached_blocks": free_cached_blocks,
            "free_uncached_blocks": free_uncached_blocks,
            "stored_block_count_total": stored_block_count_total,
            "removed_block_count_total": removed_block_count_total,
            "resident_usage_pct": resident_usage_pct,
        }

    async def _should_save_waiting_session(self) -> bool:
        if self._active_only_hbm_target() is not None:
            return True

        snapshots = await self._collect_engine_core_kv_snapshots()
        if not snapshots:
            return False
        free_uncached_blocks = min(int(snapshot.get("free_uncached_block_count", 0)) for snapshot in snapshots)
        return self._update_agentic_kv_pressure(free_uncached_blocks)

    async def _maybe_log_agentic_kv_unload(
        self,
        *,
        session_id: Optional[str],
        request_id: Optional[str],
        resume_point_id: Optional[str],
        wait_reason: Optional[str],
        before_snapshots: list[dict],
    ) -> None:
        before_summary = self._summarize_agentic_kv_snapshots(before_snapshots)
        if before_summary is None:
            return

        after_snapshots = await self._collect_engine_core_kv_snapshots()
        after_summary = self._summarize_agentic_kv_snapshots(after_snapshots)
        if after_summary is None:
            return

        unloaded_blocks = max(
            int(after_summary["removed_block_count_total"]) - int(before_summary["removed_block_count_total"]),
            0,
        )
        if unloaded_blocks <= 0:
            return

        stored_blocks = max(
            int(after_summary["stored_block_count_total"]) - int(before_summary["stored_block_count_total"]),
            0,
        )
        logger.info(
            "AgenticKV unload committed: session=%s, request=%s, resume_point=%s, wait_reason=%s, "
            "unloaded %d GPU KV blocks, stored %d LMCache blocks, cold free %d -> %d, cached free %d -> %d, resident %.1f%% -> %.1f%%",
            session_id,
            request_id,
            resume_point_id,
            wait_reason,
            unloaded_blocks,
            stored_blocks,
            int(before_summary["free_uncached_blocks"]),
            int(after_summary["free_uncached_blocks"]),
            int(before_summary["free_cached_blocks"]),
            int(after_summary["free_cached_blocks"]),
            float(before_summary["resident_usage_pct"]),
            float(after_summary["resident_usage_pct"]),
        )

    async def _evict_cached_free_blocks(self, target_free_cached_blocks: int) -> dict:
        if not hasattr(self.model, "call_engine_core_utility"):
            return {
                "available": False,
                "evicted_blocks": 0,
                "free_cached_before": 0,
                "free_cached_after": 0,
                "target_free_cached_blocks": target_free_cached_blocks,
            }

        raw_result = await self.model.call_engine_core_utility(
            "roll_evict_cached_free_blocks",
            target_free_cached_blocks,
        )
        results = self._normalize_engine_snapshots(raw_result)
        evicted_blocks = 0
        free_cached_before = 0
        free_cached_after = 0
        available = False
        for result in results:
            if not result.get("available", False):
                continue
            available = True
            evicted_blocks += int(result.get("evicted_blocks", 0) or 0)
            free_cached_before += int(result.get("free_cached_before", 0) or 0)
            free_cached_after += int(result.get("free_cached_after", 0) or 0)
        return {
            "available": available,
            "evicted_blocks": evicted_blocks,
            "free_cached_before": free_cached_before,
            "free_cached_after": free_cached_after,
            "target_free_cached_blocks": target_free_cached_blocks,
        }

    async def generate(self, batch: DataProto, generation_config) -> torch.Tensor:
        # Check if beam search is requested
        if self._should_use_beam_search(generation_config):
            return await self._generate_with_beam_search(batch, generation_config)
        else:
            return await self._generate_standard(batch, generation_config)

    def _should_use_beam_search(self, generation_config) -> bool:
        """Check if beam search should be used based on generation_config."""
        return generation_config.get("num_beams", 1) > 1 or generation_config.get("use_beam_search", False)

    async def _generate_standard(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        """Standard generate method for non-beam search cases."""
        sampling_params = SamplingParams(**create_sampling_params_for_vllm(gen_kwargs=generation_config))

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        if "multi_modal_data" in batch.non_tensor_batch:
            prompts = [TokensPrompt(data) for data in batch.non_tensor_batch["multi_modal_data"]]
        else:
            prompts = [TokensPrompt(prompt_token_ids=prompt)
                for prompt in gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            ]

        lora_request = None
        if self.is_lora:
            lora_int_ids = list(await self.model.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_request = LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path")

        async def _generate(prompt):
            request_id = random_uuid()
            result_generator = self.model.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )
            output: Optional[RequestOutput] = None
            async for result in result_generator:
                output = result
            return output

        vllm_outputs = await asyncio.gather(*[_generate(prompt) for prompt in prompts])

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=vllm_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )

        return output

    async def _generate_with_beam_search(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        """Generate using beam search method."""
        # Create beam search parameters
        beam_params = BeamSearchParams(
            beam_width=generation_config.get("num_beams", 1),
            max_tokens=generation_config.get("max_new_tokens", 50),
            temperature=generation_config.get("temperature", 0.0),
            ignore_eos=generation_config.get("ignore_eos", False),
            length_penalty=generation_config.get("length_penalty", 1.0),
            include_stop_str_in_output=generation_config.get("include_stop_str_in_output", False),
        )

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        # Prepare prompts for beam_search
        if "multi_modal_data" in batch.non_tensor_batch:
            # For multimodal data, we need to handle it differently
            # This is a simplified approach - may need refinement based on actual multimodal format
            prompts = batch.non_tensor_batch["multi_modal_data"]
        else:
            # Convert to token lists format expected by beam_search
            token_lists = gather_unpadded_input_ids(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Convert to TokensPrompt format expected by vLLM beam_search
            prompts = [{"prompt_token_ids": token_ids} for token_ids in token_lists]

        # Call beam_search method
        async def _beam_search(prompt):
            request_id = random_uuid()
            result_generator = self.model.beam_search(
                prompt=prompt,
                request_id=request_id,
                params=beam_params,
            )
            output: Optional[RequestOutput] = None
            async for result in result_generator:
                output = result
            return output

        beam_search_outputs = await asyncio.gather(*[_beam_search(prompt) for prompt in prompts])

        generated_token_ids = []
        for request_output in beam_search_outputs:
            for completion_output in request_output.outputs:
                generated_tokens = completion_output.token_ids
                generated_token_ids.append(torch.tensor(generated_tokens, device=input_ids.device))

        # Pad the sequences
        output_ids = pad_sequence(generated_token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Concatenate input and output
        output = concatenate_input_and_output(
            input_ids=input_ids,
            output_ids=output_ids,
            num_return_sequences=beam_params.beam_width
        )

        return output

    async def generate_request(self, payload: Dict):
        if "multi_modal_data" in payload:
            multi_modal_data = payload["multi_modal_data"]
            prompt_token_ids = multi_modal_data["prompt_token_ids"]
            multi_modal_data = (multi_modal_data["multi_modal_data"]
                                if "multi_modal_data" in multi_modal_data else None)
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, multi_modal_data=multi_modal_data)
        else:
            prompt = TokensPrompt(prompt_token_ids=payload["input_ids"])

        lora_request = None
        if self.is_lora:
            lora_int_ids = list(await self.model.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_request = LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path")

        trace_step = self._activate_trace_step(payload.get("trace_step"))
        try:
            sampling_params_payload = dict(payload["sampling_params"])
            agentic_kv = dict(payload.get("agentic_kv") or {})
            kv_transfer_params = dict(agentic_kv.get("request_configs") or {})
            save_on_wait = False
            before_wait_snapshots: list[dict] = []
            active_only_hbm_target = self._active_only_hbm_target()
            if self._agentic_kv_enabled() and agentic_kv:
                if active_only_hbm_target is not None:
                    save_on_wait = True
                else:
                    before_wait_snapshots = await self._collect_engine_core_kv_snapshots()
                    if before_wait_snapshots:
                        free_uncached_blocks = min(
                            int(snapshot.get("free_uncached_block_count", 0)) for snapshot in before_wait_snapshots
                        )
                        save_on_wait = self._update_agentic_kv_pressure(free_uncached_blocks)
                kv_transfer_params["lmcache.skip_save"] = not save_on_wait
            if kv_transfer_params:
                sampling_params_payload = self._merge_kv_transfer_params(
                    sampling_params_payload,
                    kv_transfer_params,
                )
            result_generator = self.model.generate(
                prompt=prompt,
                sampling_params=SamplingParams(**sampling_params_payload),
                request_id=payload["rid"],
                lora_request=lora_request,
            )
            output: Optional[RequestOutput] = None
            # vLLM support partial rollout in v1 from 0.10.1, and will return finished output
            # with finish_reason setted no matter what RequestOutputKind is.
            # For compatibility, the following except block are only for v0 and older version of v1.
            try:
                async for result in result_generator:
                    output = result
            except asyncio.CancelledError:
                if output is None:
                    return {"finish_reasons": ["abort"]}
        finally:
            self._deactivate_trace_step(trace_step)

        output_token_ids, finish_reasons, logprobs = [], [], []
        phase_timing = self._extract_phase_timing(output)
        for completion_output in output.outputs:
            output_token_ids.append(completion_output.token_ids)
            # For compatibility, older version may return unfinished result, set finish_reason of those to 'abort'.
            finish_reason = "abort" if completion_output.finish_reason is None else completion_output.finish_reason
            finish_reasons.append(finish_reason)
            if completion_output.logprobs is not None:
                logprobs.append(
                    [
                        float(lps[token_id].logprob)
                        for token_id, lps in zip(completion_output.token_ids, completion_output.logprobs)
                    ]
                )
        response = {
            "output_token_ids": output_token_ids,
            "finish_reasons": finish_reasons,
            "output_logprobs": logprobs,
            "vllm_phase_timing": phase_timing,
        }
        if self._agentic_kv_enabled() and agentic_kv:
            evicted_cached_free_blocks = 0
            if (
                save_on_wait
                and agentic_kv.get("boundary_kind") == "request_end_tool_call"
                and "stop" in finish_reasons
            ):
                if active_only_hbm_target is not None:
                    eviction_summary = await self._evict_cached_free_blocks(active_only_hbm_target)
                    evicted_cached_free_blocks = int(eviction_summary.get("evicted_blocks", 0) or 0)
                    if evicted_cached_free_blocks > 0:
                        logger.info(
                            "AgenticKV active-only eviction committed: session=%s, request=%s, resume_point=%s, "
                            "evicted %d cached-free GPU KV blocks, cached free %d -> %d, target=%d",
                            agentic_kv.get("session_id"),
                            payload["rid"],
                            agentic_kv.get("candidate_resume_point_id"),
                            evicted_cached_free_blocks,
                            int(eviction_summary.get("free_cached_before", 0) or 0),
                            int(eviction_summary.get("free_cached_after", 0) or 0),
                            int(eviction_summary.get("target_free_cached_blocks", active_only_hbm_target) or 0),
                        )
                elif before_wait_snapshots:
                    await self._maybe_log_agentic_kv_unload(
                        session_id=agentic_kv.get("session_id"),
                        request_id=payload["rid"],
                        resume_point_id=agentic_kv.get("candidate_resume_point_id"),
                        wait_reason=agentic_kv.get("wait_reason", "tool_wait"),
                        before_snapshots=before_wait_snapshots,
                    )
            response["agentic_kv"] = {
                "session_id": agentic_kv.get("session_id"),
                "save_on_wait": save_on_wait,
                "evicted_cached_free_blocks": evicted_cached_free_blocks,
            }
        elif agentic_kv:
            response["agentic_kv"] = dict(agentic_kv)
        return response

    async def abort_requests(self, request_ids):
        for id in request_ids:
            await self.model.abort(request_id=id)

    async def _reset_engine_core_lmcache(self, reason: str, *, force: bool = False) -> None:
        if not self._lmcache_enabled:
            return
        if self._engine_core_lmcache_invalidated and not force:
            return
        if not hasattr(self.model, "call_engine_core_utility"):
            return
        try:
            await self.model.call_engine_core_utility("roll_reset_lmcache_engine", reason)
            self._engine_core_lmcache_invalidated = True
        except Exception as exc:
            if not self._lmcache_reset_warning_emitted:
                logger.warning(f"Failed to reset engine-core LMCache connector: {exc}")
                self._lmcache_reset_warning_emitted = True

    # offload/reload 接口
    async def load_states(self, *args, **kwargs):
        await self.model.reset_prefix_cache()
        if not self.is_model_in_gpu:
            await self.model.load_states()
            self.is_model_in_gpu = True
        self._engine_core_lmcache_invalidated = False

    async def offload_states(self, include=None, non_blocking=False):
        await self.model.reset_prefix_cache()
        await self._reset_engine_core_lmcache("offload_states", force=True)
        if include is None or OffloadStateType.model_params in include:
            if self.is_model_in_gpu and self.worker.pipeline_config.is_actor_infer_colocated:
                await self.model.offload_states(self.sleep_level)
                self.is_model_in_gpu = False
        self._engine_core_lmcache_invalidated = False
        gc.collect()
        current_platform.empty_cache()
    
    async def process_weights_after_loading(self,*args, **kwargs):
        await self._reset_engine_core_lmcache("weight update")
        await self.model.process_weights_after_loading()
        self._engine_core_lmcache_invalidated = False

    # 参数同步相关接口
    async def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend=None):
        logger.info(f"setup_collective_group {group_name=}")
        backend = backend if backend is not None else current_platform.communication_backend
        await self.model.setup_collective_group(master_address, master_port, rank_offset, world_size, group_name, backend)

    async def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        await self._reset_engine_core_lmcache("weight update")
        await self.model.broadcast_parameter(names, dtypes, shapes, group_name, is_lora)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        await self._reset_engine_core_lmcache("weight update")
        await self.model.update_parameter_in_bucket(serialized_named_tensors, is_lora)

    async def add_lora(self, peft_config):
        peft_config["target_modules"] = set(self.worker_config.model_args.lora_target)
        await self._reset_engine_core_lmcache("add_lora")
        await self.model.add_lora(peft_config)

    async def _collect_metrics_snapshot(self):
        """Collect metrics snapshots periodically in a background thread."""
        from vllm.v1.metrics.reader import get_metrics_snapshot
        while True:
            raw_metrics = get_metrics_snapshot()
            engine_snapshots = await self._collect_engine_core_kv_snapshots()
            snapshot = {}
            timestamp_ns = time.time_ns()
            active_trace_steps = tuple(self._active_trace_steps.keys())
            tracer = get_trace_manager(component=f"{self.worker.worker_name}_metrics")
            have_real_engine_snapshot = bool(engine_snapshots)
            for engine_snapshot in engine_snapshots:
                attrs = {
                    "engine": engine_snapshot.get("engine"),
                    "source": "engine_core",
                }
                for field_name, spec in _VLLM_REALTIME_SAMPLE_SPECS.items():
                    if field_name in {"stored_block_count_delta", "removed_block_count_delta", "cleared_event_count_delta"}:
                        value = float((engine_snapshot.get("kv_events") or {}).get(field_name, 0.0))
                    else:
                        if field_name not in engine_snapshot:
                            continue
                        value = float(engine_snapshot[field_name])
                    snapshot.setdefault(spec["snapshot_key"], []).append(value)
                    if active_trace_steps:
                        for step in active_trace_steps:
                            tracer.record_sample(
                                spec["sample_name"],
                                value,
                                unit=spec["unit"],
                                step=step,
                                timestamp_ns=timestamp_ns,
                                attrs=attrs,
                            )
            for metric in raw_metrics:
                if have_real_engine_snapshot and metric.name in (
                    "vllm:num_requests_waiting",
                    "vllm:num_requests_running",
                    "vllm:num_preemptions",
                    "vllm:prefix_cache_queries",
                    "vllm:prefix_cache_hits",
                ):
                    continue
                if metric.name in _VLLM_GAUGE_METRIC_SPECS:
                    spec = _VLLM_GAUGE_METRIC_SPECS[metric.name]
                    sample_value = _sample_metric_value(metric.name, metric.value)
                    snapshot.setdefault(spec["snapshot_key"], []).append(sample_value)
                    if active_trace_steps:
                        attrs = _metric_sample_attrs(metric)
                        for step in active_trace_steps:
                            tracer.record_sample(
                                spec["sample_name"],
                                sample_value,
                                unit=spec["unit"],
                                step=step,
                                timestamp_ns=timestamp_ns,
                                attrs=attrs,
                            )
                    continue

                if metric.name in _VLLM_COUNTER_METRIC_SPECS:
                    spec = _VLLM_COUNTER_METRIC_SPECS[metric.name]
                    identity = _metric_identity(metric)
                    current_value = float(metric.value)
                    previous_value = self._counter_last_values.get(identity)
                    delta_value = 0.0 if previous_value is None else max(current_value - previous_value, 0.0)
                    self._counter_last_values[identity] = current_value
                    snapshot.setdefault(spec["snapshot_key"], []).append(delta_value)
                    if active_trace_steps:
                        attrs = _metric_sample_attrs(metric)
                        for step in active_trace_steps:
                            tracer.record_sample(
                                spec["sample_name"],
                                delta_value,
                                unit=spec["unit"],
                                kind="counter",
                                step=step,
                                timestamp_ns=timestamp_ns,
                                attrs=attrs,
                            )
                    continue

                if metric.name in _VLLM_COUNTER_RATE_METRIC_SPECS:
                    spec = _VLLM_COUNTER_RATE_METRIC_SPECS[metric.name]
                    identity = _metric_identity(metric)
                    current_value = float(metric.value)
                    previous_sample = self._counter_last_samples.get(identity)
                    rate_value = 0.0
                    if previous_sample is not None:
                        previous_timestamp_ns, previous_value = previous_sample
                        elapsed_seconds = max((timestamp_ns - previous_timestamp_ns) / 1_000_000_000, 0.0)
                        if elapsed_seconds > 0:
                            rate_value = max(current_value - previous_value, 0.0) / elapsed_seconds
                    self._counter_last_samples[identity] = (timestamp_ns, current_value)
                    snapshot[spec["snapshot_key"]] = snapshot.get(spec["snapshot_key"], 0.0) + rate_value
                    continue

            for metric_name, spec in _VLLM_COUNTER_RATE_METRIC_SPECS.items():
                if spec["snapshot_key"] not in snapshot:
                    continue
                sample_value = float(snapshot[spec["snapshot_key"]])
                if active_trace_steps:
                    attrs = {
                        "source": "prometheus_counter",
                        "aggregation": "sum",
                        "metric_name": metric_name,
                    }
                    for step in active_trace_steps:
                        tracer.record_sample(
                            spec["sample_name"],
                            sample_value,
                            unit=spec["unit"],
                            step=step,
                            timestamp_ns=timestamp_ns,
                            attrs=attrs,
                        )
            self._metrics_snapshots.append(snapshot)

            await asyncio.sleep(self._metrics_snapshot_interval)

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get aggregated metrics for the time interval since last call.

        Args:
            metric_names: Optional list of specific metric names to filter

        Returns:
            Dictionary of metric names to aggregated values
        """
        if not self._metrics_snapshots:
            return {}
        metrics_snapshots = list_of_dict_to_dict_of_list(self._metrics_snapshots)
        self._metrics_snapshots.clear()
        return reduce_metrics(metrics_snapshots)


def gather_outputs_to_pad_tensor(request_outputs: List["RequestOutput"], pad_token_id, device=None) -> torch.Tensor:
    if device is None:
        device = current_platform.device_type
    token_ids_list_of_lists = [
        torch.tensor(completion_output.token_ids, device=device)
        for request_output in request_outputs
        for completion_output in request_output.outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def create_sampling_params_for_vllm(gen_kwargs, collect_unfinished=False):
    # TODO vLLM support partial rollout in v1 from 0.10.1, and do not need to set RequestOutputKind to CUMULATIVE
    output_kind = RequestOutputKind.CUMULATIVE if collect_unfinished else RequestOutputKind.FINAL_ONLY
    return dict(
        max_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        stop=gen_kwargs["stop_strings"],
        logprobs=gen_kwargs.get("logprobs", 0),
        output_kind=output_kind,
        include_stop_str_in_output=gen_kwargs.get("include_stop_str_in_output", True),
    )
