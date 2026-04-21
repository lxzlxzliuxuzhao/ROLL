from roll.pipeline.agentic.akv.runtime import AgenticKVRuntime


def test_build_request_meta_for_current_session():
    runtime = AgenticKVRuntime(enabled=True, model_identity="model-a")
    runtime.start_session(session_id="sess-1", traj_id="traj-1")

    meta = runtime.build_request_meta(
        session_id="sess-1",
        request_id="req-1",
        env_step=4,
    )

    assert meta == {
        "enabled": True,
        "session_id": "sess-1",
        "latest_resume_point_id": None,
        "candidate_resume_point_id": "sess-1:rp:1",
        "boundary_kind": "request_end_tool_call",
        "wait_reason": "tool_wait",
        "env_step": 4,
        "request_configs": {"lmcache.tag.akv_session": "sess-1"},
    }
    assert runtime.get_session("sess-1").current_request_id == "req-1"


def test_complete_request_turns_tool_call_stop_into_waiting_external():
    runtime = AgenticKVRuntime(enabled=True, model_identity="model-a")
    runtime.start_session(session_id="sess-2", traj_id="traj-2")
    meta = runtime.build_request_meta(
        session_id="sess-2",
        request_id="req-2",
        env_step=5,
    )

    result = runtime.complete_request(
        session_id="sess-2",
        request_id="req-2",
        env_step=5,
        finish_reasons=["stop"],
        tool_names=["search_file_content"],
        candidate_resume_point_id=meta["candidate_resume_point_id"],
    )

    assert result == {
        "session_id": "sess-2",
        "state": "waiting_external",
        "resume_point_id": "sess-2:rp:1",
        "wait_reason": "tool_wait",
    }


def test_build_request_meta_supports_explicit_resume_point_id():
    runtime = AgenticKVRuntime(enabled=True, model_identity="model-a")
    runtime.start_session(session_id="sess-3", traj_id="traj-3")
    first_meta = runtime.build_request_meta(
        session_id="sess-3",
        request_id="req-3a",
        env_step=1,
    )
    runtime.complete_request(
        session_id="sess-3",
        request_id="req-3a",
        env_step=1,
        finish_reasons=["stop"],
        tool_names=["glob"],
        candidate_resume_point_id=first_meta["candidate_resume_point_id"],
    )

    resumed_meta = runtime.build_request_meta(
        session_id="sess-3",
        request_id="req-3b",
        env_step=2,
        resume_point_id="sess-3:rp:1",
    )

    assert resumed_meta["latest_resume_point_id"] == "sess-3:rp:1"
    assert resumed_meta["candidate_resume_point_id"] == "sess-3:rp:2"
    assert runtime.get_session("sess-3").current_request_id == "req-3b"


def test_finish_session_marks_session_finished():
    runtime = AgenticKVRuntime(enabled=True, model_identity="model-a")
    runtime.start_session(session_id="sess-4", traj_id="traj-4")

    runtime.finish_session("sess-4")

    session = runtime.get_session("sess-4")
    assert session.state.value == "finished"
    assert session.current_request_id is None
