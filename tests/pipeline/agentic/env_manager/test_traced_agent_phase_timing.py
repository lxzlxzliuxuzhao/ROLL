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
    def __init__(self):
        self.calls = []

    def span(self, *args, **kwargs):
        return _FakeSpan()

    def record_completed_span(self, name, **kwargs):
        self.calls.append((name, kwargs))


def test_record_inference_phase_spans_uses_real_vllm_timing():
    manager = TracedAgentNativeStepEnvManager.__new__(TracedAgentNativeStepEnvManager)
    tracer = _FakeTracer()
    request_span = _FakeSpan(start_wall_ns=1_000_000_000, end_wall_ns=1_060_000_000)

    manager._record_inference_phase_spans(
        tracer,
        request_span,
        trace_attrs={"env_step": 0, "train_step": 17},
        sample_id="sample-1",
        traj_id="traj-1",
        request_id="req-1",
        prompt_tokens=128,
        output_tokens=32,
        response_preview="preview",
        phase_timing={
            "source": "vllm_v1_request_state",
            "unit": "seconds",
            "queue_time": 0.010,
            "prefill_time": 0.020,
            "decode_time": 0.015,
            "inference_time": 0.035,
            "e2e_time": 0.040,
            "time_to_first_token": 0.030,
        },
    )

    span_names = [name for name, _ in tracer.calls]
    assert span_names == [
        "rollout.wait_worker",
        "inference.prefill",
        "inference.decode",
        "inference.overhead",
        "inference.metrics",
    ]

    wait_attrs = tracer.calls[0][1]["attrs"]
    prefill_attrs = tracer.calls[1][1]["attrs"]
    decode_attrs = tracer.calls[2][1]["attrs"]
    overhead_attrs = tracer.calls[3][1]["attrs"]
    metrics_attrs = tracer.calls[4][1]["attrs"]

    assert [kwargs["step"] for _, kwargs in tracer.calls] == [17, 17, 17, 17, 17]
    assert wait_attrs["source"] == "vllm_v1_request_state"
    assert prefill_attrs["source"] == "vllm_v1_request_state"
    assert decode_attrs["source"] == "vllm_v1_request_state"
    assert overhead_attrs["source"] == "request_residual_local"
    assert metrics_attrs["queue_time_ms"] == 10.0
    assert metrics_attrs["prefill_time_ms"] == 20.0
    assert metrics_attrs["decode_time_ms"] == 15.0
    assert metrics_attrs["overhead_time_ms"] == 15.0
    assert metrics_attrs["ttft_ms"] == 30.0


def test_extract_tool_call_names_supports_json_and_xml_formats():
    assert TracedAgentNativeStepEnvManager._extract_tool_call_names(
        '<tool_call>{"name": "search_file_content", "arguments": {"pattern": "foo"}}</tool_call>'
    ) == ["search_file_content"]

    assert TracedAgentNativeStepEnvManager._extract_tool_call_names(
        "<tool_call><function=glob><parameter=pattern>src/**/*.py</parameter></function></tool_call>"
    ) == ["glob"]

    assert TracedAgentNativeStepEnvManager._extract_tool_call_names(
        "<tool_call>{\"name\": \"glob\", \"arguments\": {\"pattern\": \"src/**/*.py\"}}</tool_call>"
        "<tool_call><function=search_file_content><parameter=pattern>router</parameter></function></tool_call>"
    ) == ["glob", "search_file_content"]


def test_build_step_trace_attrs_uses_local_env_step_and_keeps_train_step():
    manager = TracedAgentNativeStepEnvManager.__new__(TracedAgentNativeStepEnvManager)
    manager.env_config = {"env_id": 11, "tag": "RockTBNativeEnvTrain", "group_id": 0}
    manager.mode = "train"
    manager.current_step = 42

    attrs = manager._build_step_trace_attrs(
        episode_id=3,
        traj_id="RockTBNativeEnvTrain_0_3_11",
        env_step=5,
    )

    assert attrs["env_step"] == 5
    assert attrs["train_step"] == 42
    assert attrs["sample_id"] == "RockTBNativeEnvTrain_0_3_11:step:5"


def test_make_traj_id_matches_final_rollout_format():
    manager = TracedAgentNativeStepEnvManager.__new__(TracedAgentNativeStepEnvManager)
    manager.env_config = {"env_id": 11, "tag": "RockTBNativeEnvTrain", "group_id": 0}
    manager.group_seed = 19

    assert manager._make_traj_group_id(3) == "RockTBNativeEnvTrain_0_3_19"
    assert manager._make_traj_id(3) == "RockTBNativeEnvTrain_0_3_19_11"


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
    session = manager.akv_runtime.get_session("sess-abort")
    assert session.current_request_id is None
    assert session.state.value == "running"
