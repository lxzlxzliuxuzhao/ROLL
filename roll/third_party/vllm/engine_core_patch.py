from __future__ import annotations

from typing import Any


def _sum_page_size_bytes(kv_cache_manager) -> int:
    kv_cache_config = getattr(kv_cache_manager, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    total = 0
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        page_size_bytes = getattr(spec, "page_size_bytes", None)
        if page_size_bytes is not None:
            total += int(page_size_bytes)
    return int(total)


def _count_free_block_states(block_pool) -> tuple[int, int]:
    free_cached_blocks = 0
    free_uncached_blocks = 0
    for block in getattr(block_pool, "blocks", []) or []:
        if getattr(block, "is_null", False):
            continue
        ref_cnt = int(getattr(block, "ref_cnt", 0) or 0)
        if ref_cnt > 0:
            continue
        if getattr(block, "block_hash", None) is None:
            free_uncached_blocks += 1
        else:
            free_cached_blocks += 1
    return free_cached_blocks, free_uncached_blocks


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

    BlockPool.__init__ = block_pool_init

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
            _bump_counter(
                self,
                "_roll_kv_cleared_event_count_total",
                "_roll_kv_cleared_event_count_since_snapshot",
                1,
            )
        return success

    BlockPool.reset_prefix_cache = block_pool_reset_prefix_cache

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

        total_blocks = max(int(getattr(block_pool, "num_gpu_blocks", 0)) - 1, 0)
        free_blocks = int(block_pool.get_num_free_blocks())
        used_blocks = max(total_blocks - free_blocks, 0)
        cached_by_hash = getattr(block_pool, "cached_block_hash_to_block", {}) or {}
        cached_entry_count = len(cached_by_hash)
        cached_block_count = sum(len(blocks_by_id) for blocks_by_id in cached_by_hash.values())
        free_cached_blocks, free_uncached_blocks = _count_free_block_states(block_pool)
        bytes_per_block = _sum_page_size_bytes(kv_cache_manager)
        usage = float(kv_cache_manager.usage) if total_blocks else 0.0
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
            "cached_block_entries": cached_entry_count,
            "cached_block_count": cached_block_count,
            "free_cached_block_count": free_cached_blocks,
            "free_uncached_block_count": free_uncached_blocks,
            "bytes_per_block": bytes_per_block,
            "kv_cache_total_bytes": total_blocks * bytes_per_block,
            "kv_cache_free_bytes": free_blocks * bytes_per_block,
            "kv_cache_used_bytes": used_blocks * bytes_per_block,
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

    setattr(EngineCore, "roll_get_kv_cache_snapshot", roll_get_kv_cache_snapshot)
    setattr(EngineCoreProc, "roll_get_kv_cache_snapshot", roll_get_kv_cache_snapshot)
    setattr(EngineCoreProc, "_roll_kv_inspection_patched", True)
    setattr(EngineCore, "_roll_kv_inspection_patched", True)
