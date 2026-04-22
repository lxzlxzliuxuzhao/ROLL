from types import SimpleNamespace

from roll.third_party.vllm.engine_core_patch import (
    _evict_free_cached_blocks,
    patch_vllm_engine_core_inspection,
)


def test_engine_core_patch_exposes_utility_on_engine_core_proc():
    import vllm.v1.engine.core as core

    patch_vllm_engine_core_inspection()

    assert hasattr(core.EngineCore, "roll_get_kv_cache_snapshot")
    assert hasattr(core.EngineCoreProc, "roll_get_kv_cache_snapshot")
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
