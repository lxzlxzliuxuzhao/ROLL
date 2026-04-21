import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.akv.runtime import AgenticKVRuntime
from roll.pipeline.agentic.env_manager.traced_agent_native_env_manager import TracedAgentNativeStepEnvManager
from roll.utils.constants import GenerateStopReason


class _FakeSpan:
    def __init__(self, start_wall_ns: int = 1_000_000_000, end_wall_ns: int = 1_060_000_000):
        self.start_wall_ns = start_wall_ns
        self.end_wall_ns = end_wall_ns
        self.duration_ms = (end_wall_ns - start_wall_ns) / 1_000_000.0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def child_context(self, **kwargs):
        return {"child": kwargs}

    def update_attributes(self, **kwargs):
        self.updated = kwargs


class _FakeTracer:
    def span(self, *args, **kwargs):
        return _FakeSpan()

    def record_completed_span(self, name, **kwargs):
        return None


def test_make_decision_abort_releases_agentic_kv_request_binding():
    manager = TracedAgentNativeStepEnvManager.__new__(TracedAgentNativeStepEnvManager)
    manager.env_config = {"env_id": 11, "tag": "RockTBNativeEnvTrain", "group_id": 0, "max_tokens_per_step": 32}
    manager.pipeline_config = type(
        "PipelineConfig",
        (),
        {"sequence_length": 128, "parse_tool_call_parameter_to_dict": False},
    )()
    manager.worker_config = type(
        "WorkerConfig",
        (),
        {
            "generating_args": type(
                "GeneratingArgs",
                (),
                {
                    "max_new_tokens": 32,
                    "to_dict": lambda self: {"temperature": 0.0},
                },
            )(),
        },
    )()
    manager.llm_proxy = type("LLMProxy", (), {"generate": lambda self, **kwargs: None})()
    manager.logger = type("Logger", (), {"warning": lambda self, msg: None})()
    manager.current_step = 7
    manager.rollout_cache = type(
        "RolloutCacheStub",
        (),
        {
            "history": [{"observation": [{"role": "user", "content": "hello"}], "messages": []}],
            "step": 0,
        },
    )()
    manager.format_messages = lambda rollout_cache: DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            },
            batch_size=[1],
        ),
        meta_info={},
    )
    manager._record_inference_phase_spans = lambda *args, **kwargs: None
    manager.akv_runtime = AgenticKVRuntime(enabled=True, model_identity="model-a")
    manager.akv_runtime.start_session(session_id="sess-abort", traj_id="traj-abort")
    manager.rollout_cache.history[-1]["akv_session_id"] = "sess-abort"

    lm_output = manager.make_decision(
        manager.rollout_cache,
        tracer=_FakeTracer(),
        trace_attrs={"env_step": 0},
        sample_id="sample-1",
        traj_id="traj-1",
    )

    assert lm_output.meta_info["stop_reason"] == GenerateStopReason.ABORT
    assert lm_output.meta_info["agentic_kv"] == {
        "session_id": "sess-abort",
        "state": "running",
        "resume_point_id": None,
    }
    session = manager.akv_runtime.get_session("sess-abort")
    assert session.current_request_id is None
    assert session.state.value == "running"
