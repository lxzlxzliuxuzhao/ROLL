from __future__ import annotations

import time
from typing import Any

from roll.utils.tracing.core import get_trace_manager


def _sum_page_size_bytes(kv_cache_manager) -> int:
    kv_cache_config = getattr(kv_cache_manager, "kv_cache_config", None)
    if kv_cache_config is None:
        return 0

    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    kv_cache_tensors = getattr(kv_cache_config, "kv_cache_tensors", None) or []
    page_sizes = {
        int(page_size_bytes)
        for group in groups
        for page_size_bytes in [getattr(getattr(group, "kv_cache_spec", None), "page_size_bytes", None)]
        if page_size_bytes is not None
    }

    if len(page_sizes) == 1 and kv_cache_tensors:
        page_size_bytes = next(iter(page_sizes))
        return int(page_size_bytes * len(kv_cache_tensors))

    total = 0
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        page_size_bytes = getattr(spec, "page_size_bytes", None)
        if page_size_bytes is None:
            continue
        total += int(page_size_bytes) * max(len(getattr(group, "layer_names", []) or []), 1)
    return int(total)


def _sum_kv_cache_tensor_bytes(kv_cache_manager) -> int:
    kv_cache_config = getattr(kv_cache_manager, "kv_cache_config", None)
    if kv_cache_config is None:
        return 0

    kv_cache_tensors = getattr(kv_cache_config, "kv_cache_tensors", None) or []
    total = 0
    for tensor in kv_cache_tensors:
        size = getattr(tensor, "size", None)
        if size is None:
            continue
        total += int(size)
    return int(total)


def _count_free_block_states(block_pool) -> tuple[int, int]:
    free_state_counts = _scan_block_pool(block_pool)
    return free_state_counts["free_cached_blocks"], free_state_counts["free_uncached_blocks"]


def _scan_block_pool(block_pool) -> dict[str, int]:
    active_blocks = 0
    free_cached_blocks = 0
    free_uncached_blocks = 0
    cached_block_count = 0
    for block in getattr(block_pool, "blocks", []) or []:
        if getattr(block, "is_null", False):
            continue
        ref_cnt = int(getattr(block, "ref_cnt", 0) or 0)
        is_cached = getattr(block, "block_hash", None) is not None
        if is_cached:
            cached_block_count += 1
        if ref_cnt > 0:
            active_blocks += 1
        elif is_cached:
            free_cached_blocks += 1
        else:
            free_uncached_blocks += 1
    return {
        "active_blocks": active_blocks,
        "free_cached_blocks": free_cached_blocks,
        "free_uncached_blocks": free_uncached_blocks,
        "cached_block_count": cached_block_count,
    }


def _schedulable_block_total(block_pool) -> int:
    return max(int(getattr(block_pool, "num_gpu_blocks", 0)) - 1, 0)


def _usage_from_block_count(block_pool, block_count: int) -> float:
    total_blocks = _schedulable_block_total(block_pool)
    if total_blocks <= 0:
        return 0.0
    return float(max(int(block_count), 0) / total_blocks)


def _count_cached_blocks(blocks) -> int:
    return sum(
        1
        for block in blocks
        if not getattr(block, "is_null", False) and getattr(block, "block_hash", None) is not None
    )


def _evict_free_cached_blocks(block_pool, target_free_cached_blocks: int = 0) -> dict[str, int]:
    target_free_cached_blocks = max(int(target_free_cached_blocks), 0)
    free_cached_before, _ = _count_free_block_states(block_pool)
    if not getattr(block_pool, "enable_caching", False) or free_cached_before <= target_free_cached_blocks:
        return {
            "target_free_cached_blocks": target_free_cached_blocks,
            "free_cached_before": free_cached_before,
            "free_cached_after": free_cached_before,
            "evicted_blocks": 0,
        }

    evicted_blocks = 0
    remaining_cached = free_cached_before
    for block in getattr(block_pool, "blocks", []) or []:
        if remaining_cached <= target_free_cached_blocks:
            break
        if getattr(block, "is_null", False):
            continue
        if int(getattr(block, "ref_cnt", 0) or 0) != 0:
            continue
        if getattr(block, "block_hash", None) is None:
            continue
        if block_pool._maybe_evict_cached_block(block):
            evicted_blocks += 1
            remaining_cached -= 1
            if hasattr(block_pool, "_roll_cached_free_block_count"):
                setattr(
                    block_pool,
                    "_roll_cached_free_block_count",
                    max(int(getattr(block_pool, "_roll_cached_free_block_count", 0) or 0) - 1, 0),
                )

    free_cached_after, _ = _count_free_block_states(block_pool)
    return {
        "target_free_cached_blocks": target_free_cached_blocks,
        "free_cached_before": free_cached_before,
        "free_cached_after": free_cached_after,
        "evicted_blocks": evicted_blocks,
    }


def _int_attr(obj: Any, attr: str) -> int:
    return int(getattr(obj, attr, 0) or 0)


def _set_counter(obj: Any, attr: str, value: int = 0) -> None:
    if not hasattr(obj, attr):
        setattr(obj, attr, int(value))


def _bump_counter(obj: Any, total_attr: str, delta_attr: str, amount: int) -> None:
    amount = int(amount or 0)
    if amount <= 0:
        return
    setattr(obj, total_attr, _int_attr(obj, total_attr) + amount)
    setattr(obj, delta_attr, _int_attr(obj, delta_attr) + amount)


def _take_delta(obj: Any, attr: str) -> int:
    value = _int_attr(obj, attr)
    setattr(obj, attr, 0)
    return value


def _bump_int_attr(obj: Any, attr: str, amount: int) -> None:
    setattr(obj, attr, _int_attr(obj, attr) + int(amount))


def _record_logger_throughput_samples(stat_logger, prompt_throughput: float, generation_throughput: float) -> None:
    active_trace_steps = tuple(getattr(stat_logger, "_roll_active_trace_steps", ()) or ())
    if not active_trace_steps:
        return

    trace_component = getattr(stat_logger, "_roll_trace_component", None)
    tracer = get_trace_manager(component=trace_component)
    timestamp_ns = time.time_ns()
    attrs = {
        "source": "logger_manager",
        "engine": str(getattr(stat_logger, "engine_index", "0")),
    }
    for step in active_trace_steps:
        tracer.record_sample(
            "vllm.prompt_throughput_tps",
            float(prompt_throughput),
            unit="tok/s",
            step=int(step),
            timestamp_ns=timestamp_ns,
            attrs=attrs,
        )
        tracer.record_sample(
            "vllm.generation_throughput_tps",
            float(generation_throughput),
            unit="tok/s",
            step=int(step),
            timestamp_ns=timestamp_ns,
            attrs=attrs,
        )


def _run_engine_core_with_roll_patch(*args, **kwargs):
    patch_vllm_engine_core_inspection()

    from vllm.v1.engine.core import EngineCoreProc

    original = getattr(EngineCoreProc, "_roll_original_run_engine_core", None)
    if original is None:
        raise RuntimeError("Missing original EngineCoreProc.run_engine_core while bootstrapping ROLL engine-core patch.")
    return original(*args, **kwargs)


def patch_vllm_engine_core_inspection() -> None:
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    from vllm.v1.engine import EngineCoreEventType
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.metrics import loggers as metric_loggers
    from vllm.v1.request import Request

    if getattr(EngineCore, "_roll_kv_inspection_patched", False):
        return

    if not hasattr(EngineCoreProc, "_roll_original_run_engine_core"):
        setattr(EngineCoreProc, "_roll_original_run_engine_core", EngineCoreProc.run_engine_core)
    EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_with_roll_patch)

    original_scheduler_init = Scheduler.__init__

    def scheduler_init(self, *args, **kwargs):
        original_scheduler_init(self, *args, **kwargs)
        _set_counter(self, "_roll_num_preemptions_total")
        _set_counter(self, "_roll_num_preemptions_since_snapshot")

    Scheduler.__init__ = scheduler_init

    original_scheduler_add_request = Scheduler.add_request

    def scheduler_add_request(self, request: Request) -> None:
        setattr(request, "_roll_scheduler", self)
        original_scheduler_add_request(self, request)

    Scheduler.add_request = scheduler_add_request

    original_request_record_event = Request.record_event

    def request_record_event(self, event_type, timestamp=None) -> None:
        original_request_record_event(self, event_type, timestamp)
        if event_type == EngineCoreEventType.PREEMPTED:
            scheduler = getattr(self, "_roll_scheduler", None)
            if scheduler is not None:
                _bump_counter(
                    scheduler,
                    "_roll_num_preemptions_total",
                    "_roll_num_preemptions_since_snapshot",
                    1,
                )

    Request.record_event = request_record_event

    original_kv_cache_manager_init = KVCacheManager.__init__

    def kv_cache_manager_init(self, *args, **kwargs):
        original_kv_cache_manager_init(self, *args, **kwargs)
        for attr in (
            "_roll_prefix_cache_queries_total",
            "_roll_prefix_cache_queries_since_snapshot",
            "_roll_prefix_cache_hits_total",
            "_roll_prefix_cache_hits_since_snapshot",
            "_roll_prefix_cache_resets_total",
            "_roll_prefix_cache_resets_since_snapshot",
        ):
            _set_counter(self, attr)

    KVCacheManager.__init__ = kv_cache_manager_init

    original_get_computed_blocks = KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request):
        computed_blocks, num_new_computed_tokens = original_get_computed_blocks(self, request)
        sampling_params = getattr(request, "sampling_params", None)
        should_track_prefix_query = bool(
            getattr(self, "enable_caching", False)
            and not (sampling_params is not None and getattr(sampling_params, "prompt_logprobs", None) is not None)
        )
        if should_track_prefix_query:
            request_tokens = int(getattr(request, "num_tokens", 0) or 0)
            _bump_counter(
                self,
                "_roll_prefix_cache_queries_total",
                "_roll_prefix_cache_queries_since_snapshot",
                request_tokens,
            )
            _bump_counter(
                self,
                "_roll_prefix_cache_hits_total",
                "_roll_prefix_cache_hits_since_snapshot",
                int(num_new_computed_tokens or 0),
            )
        return computed_blocks, num_new_computed_tokens

    KVCacheManager.get_computed_blocks = get_computed_blocks

    original_reset_prefix_cache = KVCacheManager.reset_prefix_cache

    def reset_prefix_cache(self) -> bool:
        success = original_reset_prefix_cache(self)
        if success:
            _bump_counter(
                self,
                "_roll_prefix_cache_resets_total",
                "_roll_prefix_cache_resets_since_snapshot",
                1,
            )
        return success

    KVCacheManager.reset_prefix_cache = reset_prefix_cache

    original_block_pool_init = BlockPool.__init__

    def block_pool_init(self, *args, **kwargs):
        original_block_pool_init(self, *args, **kwargs)
        for attr in (
            "_roll_kv_stored_event_count_total",
            "_roll_kv_stored_event_count_since_snapshot",
            "_roll_kv_removed_event_count_total",
            "_roll_kv_removed_event_count_since_snapshot",
            "_roll_kv_cleared_event_count_total",
            "_roll_kv_cleared_event_count_since_snapshot",
            "_roll_kv_stored_block_count_total",
            "_roll_kv_stored_block_count_since_snapshot",
            "_roll_kv_removed_block_count_total",
            "_roll_kv_removed_block_count_since_snapshot",
        ):
            _set_counter(self, attr)
        _set_counter(self, "_roll_cached_free_block_count")

    BlockPool.__init__ = block_pool_init

    def get_num_cached_free_blocks(self) -> int:
        return _int_attr(self, "_roll_cached_free_block_count")

    def get_cached_free_usage(self) -> float:
        return _usage_from_block_count(self, get_num_cached_free_blocks(self))

    BlockPool.get_num_cached_free_blocks = get_num_cached_free_blocks
    BlockPool.get_cached_free_usage = get_cached_free_usage

    def get_new_blocks(self, num_blocks: int):
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret = self.free_block_queue.popleft_n(num_blocks)

        if self.enable_caching:
            _bump_int_attr(self, "_roll_cached_free_block_count", -_count_cached_blocks(ret))
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    BlockPool.get_new_blocks = get_new_blocks

    def touch(self, blocks) -> None:
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                if block.ref_cnt == 0 and not block.is_null:
                    if block.block_hash is not None:
                        _bump_int_attr(self, "_roll_cached_free_block_count", -1)
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

    BlockPool.touch = touch

    def free_blocks(self, ordered_blocks) -> None:
        blocks_list = list(ordered_blocks)
        cached_free_blocks_added = 0
        for block in blocks_list:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null and block.block_hash is not None:
                cached_free_blocks_added += 1
        self.free_block_queue.append_n([
            block for block in blocks_list if block.ref_cnt == 0 and not block.is_null
        ])
        if cached_free_blocks_added:
            _bump_int_attr(self, "_roll_cached_free_block_count", cached_free_blocks_added)

    BlockPool.free_blocks = free_blocks

    original_cache_full_blocks = BlockPool.cache_full_blocks

    def cache_full_blocks(
        self,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
    ) -> None:
        original_cache_full_blocks(
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
        )
        newly_cached_blocks = max(int(num_full_blocks) - int(num_cached_blocks), 0)
        if newly_cached_blocks > 0:
            _bump_counter(
                self,
                "_roll_kv_stored_event_count_total",
                "_roll_kv_stored_event_count_since_snapshot",
                1,
            )
            _bump_counter(
                self,
                "_roll_kv_stored_block_count_total",
                "_roll_kv_stored_block_count_since_snapshot",
                newly_cached_blocks,
            )

    BlockPool.cache_full_blocks = cache_full_blocks

    original_maybe_evict_cached_block = BlockPool._maybe_evict_cached_block

    def maybe_evict_cached_block(self, block) -> bool:
        evicted = original_maybe_evict_cached_block(self, block)
        if evicted:
            _bump_counter(
                self,
                "_roll_kv_removed_event_count_total",
                "_roll_kv_removed_event_count_since_snapshot",
                1,
            )
            _bump_counter(
                self,
                "_roll_kv_removed_block_count_total",
                "_roll_kv_removed_block_count_since_snapshot",
                1,
            )
        return evicted

    BlockPool._maybe_evict_cached_block = maybe_evict_cached_block

    original_block_pool_reset_prefix_cache = BlockPool.reset_prefix_cache

    def block_pool_reset_prefix_cache(self) -> bool:
        success = original_block_pool_reset_prefix_cache(self)
        if success:
            setattr(self, "_roll_cached_free_block_count", 0)
            _bump_counter(
                self,
                "_roll_kv_cleared_event_count_total",
                "_roll_kv_cleared_event_count_since_snapshot",
                1,
            )
        return success

    BlockPool.reset_prefix_cache = block_pool_reset_prefix_cache

    original_scheduler_make_stats = Scheduler.make_stats

    def scheduler_make_stats(self, *args, **kwargs):
        stats = original_scheduler_make_stats(self, *args, **kwargs)
        if stats is None:
            return None

        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return stats

        total_blocks = _schedulable_block_total(block_pool)
        if total_blocks <= 0:
            active_usage = 0.0
            cached_free_usage = 0.0
            resident_usage = 0.0
        else:
            active_usage = float(getattr(stats, "kv_cache_usage", 0.0) or 0.0)
            if hasattr(block_pool, "get_cached_free_usage"):
                cached_free_usage = float(block_pool.get_cached_free_usage())
            else:
                cached_free_usage = _usage_from_block_count(
                    block_pool,
                    int(
                        block_pool.get_num_cached_free_blocks()
                        if hasattr(block_pool, "get_num_cached_free_blocks")
                        else _count_free_block_states(block_pool)[0]
                    ),
                )
            resident_usage = min(active_usage + cached_free_usage, 1.0)

        setattr(stats, "kv_cache_active_usage", active_usage)
        setattr(stats, "kv_cache_cached_free_usage", cached_free_usage)
        setattr(stats, "kv_cache_resident_usage", resident_usage)
        setattr(
            stats,
            "kv_cache_cached_free_blocks",
            int(
                block_pool.get_num_cached_free_blocks()
                if hasattr(block_pool, "get_num_cached_free_blocks")
                else _count_free_block_states(block_pool)[0]
            ),
        )
        setattr(stats, "kv_cache_total_schedulable_blocks", total_blocks)
        return stats

    Scheduler.make_stats = scheduler_make_stats

    def logging_stat_logger_log(self):
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(self.num_generation_tokens, now)

        self._reset(now)

        scheduler_stats = self.last_scheduler_stats

        log_fn = metric_loggers.logger.info
        if not any(
            (
                prompt_throughput,
                generation_throughput,
                self.last_prompt_throughput,
                self.last_generation_throughput,
            )
        ):
            log_fn = metric_loggers.logger.debug
        self.last_generation_throughput = generation_throughput
        self.last_prompt_throughput = prompt_throughput
        _record_logger_throughput_samples(self, prompt_throughput, generation_throughput)

        engine_snapshot = dict(getattr(self, "_roll_engine_core_snapshot", {}) or {})
        running_reqs = int(engine_snapshot.get("num_requests_running", scheduler_stats.num_running_reqs))
        waiting_reqs = int(engine_snapshot.get("num_requests_waiting", scheduler_stats.num_waiting_reqs))
        active_usage = float(
            engine_snapshot.get(
                "kv_cache_active_usage_perc",
                float(getattr(scheduler_stats, "kv_cache_active_usage", scheduler_stats.kv_cache_usage)) * 100.0,
            )
        )
        cached_free_usage = float(
            engine_snapshot.get(
                "kv_cache_cached_free_usage_perc",
                float(getattr(scheduler_stats, "kv_cache_cached_free_usage", 0.0)) * 100.0,
            )
        )
        resident_usage = float(
            engine_snapshot.get(
                "kv_cache_resident_usage_perc",
                float(getattr(scheduler_stats, "kv_cache_resident_usage", scheduler_stats.kv_cache_usage)) * 100.0,
            )
        )
        cached_block_count = engine_snapshot.get("cached_block_count")
        free_cached_blocks = engine_snapshot.get("free_cached_block_count")
        free_uncached_blocks = engine_snapshot.get("free_uncached_block_count")

        if cached_block_count is not None and free_cached_blocks is not None and free_uncached_blocks is not None:
            log_fn(
                "Engine %03d: "
                "Avg prompt throughput: %.1f tokens/s, "
                "Avg generation throughput: %.1f tokens/s, "
                "Running: %d reqs, Waiting: %d reqs, "
                "GPU KV cache usage: active %.1f%%, cached free %.1f%%, resident %.1f%%, "
                "Cached blocks: %d, Free cached blocks: %d, Cold free blocks: %d, "
                "Prefix cache hit rate: %.1f%%",
                self.engine_index,
                prompt_throughput,
                generation_throughput,
                running_reqs,
                waiting_reqs,
                active_usage,
                cached_free_usage,
                resident_usage,
                int(cached_block_count),
                int(free_cached_blocks),
                int(free_uncached_blocks),
                self.prefix_caching_metrics.hit_rate * 100,
            )
        else:
            log_fn(
                "Engine %03d: "
                "Avg prompt throughput: %.1f tokens/s, "
                "Avg generation throughput: %.1f tokens/s, "
                "Running: %d reqs, Waiting: %d reqs, "
                "GPU KV cache usage: active %.1f%%, cached free %.1f%%, resident %.1f%%, "
                "Prefix cache hit rate: %.1f%%",
                self.engine_index,
                prompt_throughput,
                generation_throughput,
                running_reqs,
                waiting_reqs,
                active_usage,
                cached_free_usage,
                resident_usage,
                self.prefix_caching_metrics.hit_rate * 100,
            )
        self.spec_decoding_logging.log(log_fn=log_fn)

    metric_loggers.LoggingStatLogger.log = logging_stat_logger_log

    original_prometheus_init = metric_loggers.PrometheusStatLogger.__init__

    def prometheus_stat_logger_init(self, vllm_config, engine_indexes=None):
        original_prometheus_init(self, vllm_config, engine_indexes)
        labelnames = ["model_name", "engine"]
        model_name = vllm_config.model_config.served_model_name
        engine_indexes_local = self.engine_indexes
        gauge_cached_free_usage = self._gauge_cls(
            name="vllm:kv_cache_cached_free_usage_perc",
            documentation="Free cached KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
        )
        self.gauge_kv_cache_cached_free_usage = metric_loggers.make_per_engine(
            gauge_cached_free_usage,
            engine_indexes_local,
            model_name,
        )
        gauge_resident_usage = self._gauge_cls(
            name="vllm:kv_cache_resident_usage_perc",
            documentation="Resident KV-cache usage including active and free cached blocks. 1 means 100 percent usage.",
            labelnames=labelnames,
        )
        self.gauge_kv_cache_resident_usage = metric_loggers.make_per_engine(
            gauge_resident_usage,
            engine_indexes_local,
            model_name,
        )

    metric_loggers.PrometheusStatLogger.__init__ = prometheus_stat_logger_init

    original_prometheus_record = metric_loggers.PrometheusStatLogger.record

    def prometheus_stat_logger_record(self, scheduler_stats, iteration_stats, engine_idx: int = 0):
        original_prometheus_record(self, scheduler_stats, iteration_stats, engine_idx)
        if scheduler_stats is None:
            return
        gauge_cached_free = getattr(self, "gauge_kv_cache_cached_free_usage", None)
        gauge_resident = getattr(self, "gauge_kv_cache_resident_usage", None)
        if gauge_cached_free is not None:
            gauge_cached_free[engine_idx].set(float(getattr(scheduler_stats, "kv_cache_cached_free_usage", 0.0)))
        if gauge_resident is not None:
            gauge_resident[engine_idx].set(
                float(getattr(scheduler_stats, "kv_cache_resident_usage", getattr(scheduler_stats, "kv_cache_usage", 0.0)))
            )

    metric_loggers.PrometheusStatLogger.record = prometheus_stat_logger_record

    def roll_get_kv_cache_snapshot(self) -> dict[str, Any]:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            return {"available": False, "reason": "missing_scheduler"}

        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return {"available": False, "reason": "missing_kv_cache_manager"}

        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return {"available": False, "reason": "missing_block_pool"}

        total_blocks = _schedulable_block_total(block_pool)
        block_scan = _scan_block_pool(block_pool)
        used_blocks = int(block_scan["active_blocks"])
        free_cached_blocks = int(block_scan["free_cached_blocks"])
        free_uncached_blocks = int(block_scan["free_uncached_blocks"])
        free_blocks = free_cached_blocks + free_uncached_blocks
        cached_by_hash = getattr(block_pool, "cached_block_hash_to_block", {}) or {}
        cached_entry_count = len(cached_by_hash)
        cached_block_count = int(block_scan["cached_block_count"])
        free_cached_blocks_counter = int(
            block_pool.get_num_cached_free_blocks()
            if hasattr(block_pool, "get_num_cached_free_blocks")
            else _count_free_block_states(block_pool)[0]
        )
        bytes_per_block = _sum_page_size_bytes(kv_cache_manager)
        allocated_bytes = _sum_kv_cache_tensor_bytes(kv_cache_manager)
        schedulable_bytes = total_blocks * bytes_per_block
        reserved_bytes = max(allocated_bytes - schedulable_bytes, 0)
        usage = _usage_from_block_count(block_pool, used_blocks)
        cached_free_usage = _usage_from_block_count(block_pool, free_cached_blocks)
        resident_usage = min(usage + cached_free_usage, 1.0)
        prefix_queries_delta = _take_delta(kv_cache_manager, "_roll_prefix_cache_queries_since_snapshot")
        prefix_hits_delta = _take_delta(kv_cache_manager, "_roll_prefix_cache_hits_since_snapshot")
        prefix_resets_delta = _take_delta(kv_cache_manager, "_roll_prefix_cache_resets_since_snapshot")
        preemptions_delta = _take_delta(scheduler, "_roll_num_preemptions_since_snapshot")
        kv_events = {
            "stored_event_count_delta": _take_delta(block_pool, "_roll_kv_stored_event_count_since_snapshot"),
            "removed_event_count_delta": _take_delta(block_pool, "_roll_kv_removed_event_count_since_snapshot"),
            "cleared_event_count_delta": _take_delta(block_pool, "_roll_kv_cleared_event_count_since_snapshot"),
            "stored_block_count_delta": _take_delta(block_pool, "_roll_kv_stored_block_count_since_snapshot"),
            "removed_block_count_delta": _take_delta(block_pool, "_roll_kv_removed_block_count_since_snapshot"),
            "stored_event_count_total": _int_attr(block_pool, "_roll_kv_stored_event_count_total"),
            "removed_event_count_total": _int_attr(block_pool, "_roll_kv_removed_event_count_total"),
            "cleared_event_count_total": _int_attr(block_pool, "_roll_kv_cleared_event_count_total"),
            "stored_block_count_total": _int_attr(block_pool, "_roll_kv_stored_block_count_total"),
            "removed_block_count_total": _int_attr(block_pool, "_roll_kv_removed_block_count_total"),
        }

        return {
            "available": True,
            "num_gpu_blocks_total": total_blocks,
            "num_gpu_blocks_free": free_blocks,
            "num_gpu_blocks_used": used_blocks,
            "kv_cache_usage_perc": usage * 100.0,
            "kv_cache_active_usage_perc": usage * 100.0,
            "kv_cache_cached_free_usage_perc": cached_free_usage * 100.0,
            "kv_cache_resident_usage_perc": resident_usage * 100.0,
            "cached_block_entries": cached_entry_count,
            "cached_block_count": cached_block_count,
            "free_cached_block_count": free_cached_blocks,
            "free_cached_block_count_counter": free_cached_blocks_counter,
            "free_uncached_block_count": free_uncached_blocks,
            "bytes_per_block": bytes_per_block,
            "kv_cache_total_bytes": schedulable_bytes,
            "kv_cache_allocated_bytes": allocated_bytes,
            "kv_cache_free_bytes": free_blocks * bytes_per_block,
            "kv_cache_used_bytes": used_blocks * bytes_per_block,
            "kv_cache_reserved_bytes": reserved_bytes,
            "num_requests_running": len(getattr(scheduler, "running", []) or []),
            "num_requests_waiting": len(getattr(scheduler, "waiting", []) or []),
            "num_kv_cache_groups": int(getattr(kv_cache_manager, "num_kv_cache_groups", 0) or 0),
            "prefix_cache_enabled": bool(getattr(kv_cache_manager, "enable_caching", False)),
            "prefix_cache_queries_delta": prefix_queries_delta,
            "prefix_cache_hits_delta": prefix_hits_delta,
            "prefix_cache_queries_total": _int_attr(kv_cache_manager, "_roll_prefix_cache_queries_total"),
            "prefix_cache_hits_total": _int_attr(kv_cache_manager, "_roll_prefix_cache_hits_total"),
            "prefix_cache_resets_delta": prefix_resets_delta,
            "prefix_cache_resets_total": _int_attr(kv_cache_manager, "_roll_prefix_cache_resets_total"),
            "prefix_cache_hit_rate_delta_pct": (
                (float(prefix_hits_delta) * 100.0 / float(prefix_queries_delta))
                if prefix_queries_delta > 0
                else None
            ),
            "num_preemptions_delta": preemptions_delta,
            "num_preemptions_total": _int_attr(scheduler, "_roll_num_preemptions_total"),
            "enable_kv_cache_events": bool(getattr(block_pool, "enable_kv_cache_events", False)),
            "kv_events": kv_events,
        }

    def roll_reset_lmcache_engine(self, reason: str = "roll_manual_reset") -> dict[str, Any]:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            return {"available": False, "reason": "missing_scheduler"}

        connector = None
        if hasattr(scheduler, "get_kv_connector"):
            connector = scheduler.get_kv_connector()
        if connector is None:
            connector = getattr(scheduler, "connector", None)
        if connector is None:
            return {"available": False, "reason": "missing_connector"}

        if hasattr(connector, "reset_lmcache_engine"):
            connector.reset_lmcache_engine(reason=reason)
            return {
                "available": True,
                "connector_type": type(connector).__name__,
                "reset": True,
                "reason": reason,
            }

        lmcache_engine = getattr(connector, "_lmcache_engine", None)
        if hasattr(lmcache_engine, "reset"):
            lmcache_engine.reset(reason=reason)
            return {
                "available": True,
                "connector_type": type(connector).__name__,
                "reset": True,
                "reason": reason,
            }

        return {
            "available": False,
            "reason": "reset_unsupported",
            "connector_type": type(connector).__name__,
        }

    def roll_evict_cached_free_blocks(self, target_free_cached_blocks: int = 0) -> dict[str, Any]:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            return {"available": False, "reason": "missing_scheduler"}

        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return {"available": False, "reason": "missing_kv_cache_manager"}

        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return {"available": False, "reason": "missing_block_pool"}

        summary = _evict_free_cached_blocks(block_pool, target_free_cached_blocks=target_free_cached_blocks)
        return {
            "available": True,
            **summary,
        }

    setattr(EngineCore, "roll_get_kv_cache_snapshot", roll_get_kv_cache_snapshot)
    setattr(EngineCoreProc, "roll_get_kv_cache_snapshot", roll_get_kv_cache_snapshot)
    setattr(EngineCore, "roll_reset_lmcache_engine", roll_reset_lmcache_engine)
    setattr(EngineCoreProc, "roll_reset_lmcache_engine", roll_reset_lmcache_engine)
    setattr(EngineCore, "roll_evict_cached_free_blocks", roll_evict_cached_free_blocks)
    setattr(EngineCoreProc, "roll_evict_cached_free_blocks", roll_evict_cached_free_blocks)
    setattr(EngineCoreProc, "_roll_kv_inspection_patched", True)
    setattr(EngineCore, "_roll_kv_inspection_patched", True)
