from roll.pipeline.agentic.env.rock.sandbox_manager_v2 import RunSessionResponse, RunStatus, SandboxManagerV2
from roll.pipeline.agentic.env.sandbox.rock_tb_native_env import RockTBNativeEnv


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


class _RecordedSpan:
    def __init__(self, sink, name, attrs):
        self._sink = sink
        self._name = name
        self._attrs = attrs

    def __enter__(self):
        self._sink.append((self._name, self._attrs))
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _SpanRecorder:
    def __init__(self):
        self.events = []

    def span(self, name, **attrs):
        return _RecordedSpan(self.events, name, attrs)


def _build_env():
    env = RockTBNativeEnv.__new__(RockTBNativeEnv)
    env.current_step = 0
    env.current_session_step = 0
    env.group_id = 0
    env.rollout_time = 0.0
    env.max_env_time = 10_000.0
    env.exp_mode = "train"
    env.env_timeout = False
    env.env_reset_failed = False
    env.failure_mode = ""
    env.error_messages = []
    env.reward = 0
    env.terminated = False
    env.truncated = False
    env.stop_reason = ""
    env.task_id = "task-1"
    env.task_name = "task-1"
    env.max_steps = 10
    env.session_num = 0
    env.max_multi_session_num = 1
    env.test_timeout_sec = 123
    env.debug = False
    env.is_closed = False
    env.test_output = ""
    env.logger = _FakeLogger()
    env.test_files = []
    return env


def test_step_emits_terminal_env_subphase_spans(monkeypatch):
    env = _build_env()
    recorder = _SpanRecorder()

    class _FakeSandboxManager:
        def __init__(self):
            self.stop_calls = 0
            self.timeout_seen = None

        def format_response_payload(self, response):
            return "payload", {"action_is_valid": True}

        def fetch_agent_request(self, index, response_payload=None):
            return RunSessionResponse(exit_code=0, output="SESSION_END")

        def run_tests(self, test_files, test_timeout_sec, task_name):
            self.timeout_seen = test_timeout_sec
            return True, "", "All tests passed", "bash /tests/run-tests.sh"

        def stop_sandbox(self):
            self.stop_calls += 1

    env.sandbox_manager = _FakeSandboxManager()
    monkeypatch.setattr(env, "_trace_env_span", lambda name, **attrs: recorder.span(name, **attrs))

    observation, reward, terminated, truncated, info = env.step("final answer")

    span_names = [name for name, _ in recorder.events]
    assert span_names == [
        "env.format_response",
        "env.fetch_request",
        "env.check_termination",
        "env.reward_test",
        "env.close",
    ]
    reward_attrs = next(attrs for name, attrs in recorder.events if name == "env.reward_test")
    assert reward_attrs["test_timeout_sec"] == 123
    assert observation == "SESSION_END"
    assert reward is True
    assert terminated is True
    assert truncated is False
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert env.sandbox_manager.timeout_seen == 123
    assert env.sandbox_manager.stop_calls == 1


def test_reset_agent_status_emits_init_subphase_spans(monkeypatch):
    env = _build_env()
    recorder = _SpanRecorder()

    class _FakeSandboxManager:
        def start_agent(self, prompt):
            return RunSessionResponse(exit_code=0, output="agent started")

        def fetch_agent_request(self, index, response_payload=None):
            return RunSessionResponse(exit_code=0, output='{"messages": [{"role": "user", "content": "hi"}], "tools": []}')

        def get_messages_and_tools(self, request_payload):
            return [{"role": "user", "content": "hi"}], [], ""

    env.sandbox_manager = _FakeSandboxManager()
    monkeypatch.setattr(env, "_trace_env_span", lambda name, **attrs: recorder.span(name, **attrs))

    observation, tools, error_msg = env.reset_agent_status(prompt="hello")

    span_names = [name for name, _ in recorder.events]
    assert span_names == [
        "env.start_agent",
        "env.fetch_init_request",
        "env.parse_init_request",
    ]
    assert observation == [{"role": "user", "content": "hi"}]
    assert tools == []
    assert error_msg == ""


def test_run_tests_uses_configured_timeout(tmp_path):
    manager = SandboxManagerV2.__new__(SandboxManagerV2)
    manager.logger = _FakeLogger()
    manager.test_session_name = "test-session"
    manager.failure_mode = ""
    manager.error_messages = []

    task_dir = tmp_path / "task-1"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_dummy.py").write_text("def test_dummy():\n    assert True\n")
    (task_dir / "run-tests.sh").write_text("#!/usr/bin/env bash\necho All tests passed\n")

    manager.create_session = lambda session: True
    manager._compress_directory = lambda src, dst: True
    manager.upload_file = lambda src, dst: (True, "")

    def _run_in_session(command, session_name):
        if command == "pwd":
            return RunSessionResponse(exit_code=0, output="/app")
        return RunSessionResponse(exit_code=0, output="")

    manager.run_in_session = _run_in_session

    captured = {}

    def _run_session_with_timeout(session_name, command, timeout, output_file):
        captured["timeout"] = timeout
        return RunStatus.SUCCESS, "All tests passed"

    manager.run_session_with_timeout = _run_session_with_timeout
    manager._parse_test_results = lambda test_output: True

    is_resolved, error_info, test_output, test_command = manager.run_tests(
        [str(tmp_path)],
        test_timeout_sec=321,
        task_name="task-1",
    )

    assert is_resolved is True
    assert error_info == ""
    assert test_output == "All tests passed"
    assert test_command == "bash /tests/run-tests.sh"
    assert captured["timeout"] == 321
