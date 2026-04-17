from roll.third_party.vllm.engine_core_patch import patch_vllm_engine_core_inspection


def test_engine_core_patch_exposes_utility_on_engine_core_proc():
    import vllm.v1.engine.core as core

    patch_vllm_engine_core_inspection()

    assert hasattr(core.EngineCore, "roll_get_kv_cache_snapshot")
    assert hasattr(core.EngineCoreProc, "roll_get_kv_cache_snapshot")
    assert hasattr(core.EngineCoreProc, "_roll_original_run_engine_core")
    assert core.EngineCoreProc.run_engine_core is not core.EngineCoreProc._roll_original_run_engine_core
