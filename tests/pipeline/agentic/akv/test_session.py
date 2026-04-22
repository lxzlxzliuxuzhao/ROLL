import pytest

from roll.pipeline.agentic.akv import (
    FreeBlockWatermark,
    ResumeBoundaryKind,
    TrajectoryKVSession,
    TrajectoryKVSessionState,
    WaitReason,
)
from roll.pipeline.agentic.agentic_config import AgenticKVConfig


def test_tool_wait_transition_to_waiting_external():
    session = TrajectoryKVSession.create(
        session_id="sess-1",
        traj_id="traj-1",
        model_identity="model-a",
    )
    session.mark_running(request_id="req-1")

    resume_point = session.mark_waiting(
        request_id="req-1",
        resume_point_id="rp-1",
        boundary_kind=ResumeBoundaryKind.REQUEST_END_TOOL_CALL,
        wait_reason=WaitReason.TOOL_WAIT,
        env_step=3,
    )

    assert session.state == TrajectoryKVSessionState.WAITING_EXTERNAL
    assert session.current_request_id is None
    assert session.wait_reason == WaitReason.TOOL_WAIT
    assert session.latest_resume_point_id == "rp-1"
    assert session.generation_index == 1
    assert session.resume_points["rp-1"] == resume_point
    assert resume_point.request_id == "req-1"
    assert resume_point.env_step == 3


def test_waiting_session_can_finish_directly():
    session = TrajectoryKVSession.create(
        session_id="sess-2",
        traj_id="traj-2",
        model_identity="model-b",
    )
    session.mark_running(request_id="req-2")
    session.mark_waiting(
        request_id="req-2",
        resume_point_id="rp-2",
        boundary_kind=ResumeBoundaryKind.REQUEST_END,
        wait_reason=WaitReason.ENV_WAIT,
        env_step=5,
    )

    session.mark_finished()

    assert session.state == TrajectoryKVSessionState.FINISHED
    assert session.current_request_id is None
    assert session.wait_reason is None


def test_mark_waiting_requires_active_request_binding():
    session = TrajectoryKVSession.create(
        session_id="sess-3",
        traj_id="traj-3",
        model_identity="model-c",
    )

    with pytest.raises(ValueError, match="without an active request"):
        session.mark_waiting(
            request_id="req-3",
            resume_point_id="rp-3",
            boundary_kind=ResumeBoundaryKind.REQUEST_END,
            wait_reason=WaitReason.ENV_WAIT,
            env_step=0,
        )


def test_mark_waiting_rejects_mismatched_request_binding():
    session = TrajectoryKVSession.create(
        session_id="sess-4",
        traj_id="traj-4",
        model_identity="model-d",
    )
    session.mark_running(request_id="req-4a")

    with pytest.raises(ValueError, match="active request_id='req-4a'"):
        session.mark_waiting(
            request_id="req-4b",
            resume_point_id="rp-4",
            boundary_kind=ResumeBoundaryKind.REQUEST_END_TOOL_CALL,
            wait_reason=WaitReason.TOOL_WAIT,
            env_step=1,
        )


def test_mark_running_rejects_rebinding_active_request():
    session = TrajectoryKVSession.create(
        session_id="sess-4b",
        traj_id="traj-4b",
        model_identity="model-d",
    )
    session.mark_running(request_id="req-4b-a")

    with pytest.raises(ValueError, match="cannot rebind running session"):
        session.mark_running(request_id="req-4b-b")


def test_mark_waiting_rejects_duplicate_resume_point_id():
    session = TrajectoryKVSession.create(
        session_id="sess-4c",
        traj_id="traj-4c",
        model_identity="model-d",
    )
    session.mark_running(request_id="req-4c")
    session.mark_waiting(
        request_id="req-4c",
        resume_point_id="rp-dup",
        boundary_kind=ResumeBoundaryKind.REQUEST_END,
        wait_reason=WaitReason.ENV_WAIT,
        env_step=1,
    )
    session.mark_running(request_id="req-4c")

    with pytest.raises(ValueError, match="already exists"):
        session.mark_waiting(
            request_id="req-4c",
            resume_point_id="rp-dup",
            boundary_kind=ResumeBoundaryKind.REQUEST_END_TOOL_CALL,
            wait_reason=WaitReason.TOOL_WAIT,
            env_step=2,
        )


def test_terminal_states_reject_rewrites():
    finished_session = TrajectoryKVSession.create(
        session_id="sess-5",
        traj_id="traj-5",
        model_identity="model-e",
    )
    finished_session.mark_finished()
    with pytest.raises(ValueError, match="terminal session state finished"):
        finished_session.mark_finished()
    with pytest.raises(ValueError, match="terminal session state finished"):
        finished_session.mark_invalidated("reason")

    invalidated_session = TrajectoryKVSession.create(
        session_id="sess-6",
        traj_id="traj-6",
        model_identity="model-f",
    )
    invalidated_session.mark_invalidated("boom")
    with pytest.raises(ValueError, match="terminal session state invalidated"):
        invalidated_session.mark_finished()
    with pytest.raises(ValueError, match="terminal session state invalidated"):
        invalidated_session.mark_invalidated("again")


def test_free_block_watermark_hysteresis():
    watermark = FreeBlockWatermark(low=10, high=20)

    unloading = False
    unloading = watermark.update(free_blocks=15, unloading=unloading)
    assert unloading is False

    unloading = watermark.update(free_blocks=9, unloading=unloading)
    assert unloading is True

    unloading = watermark.update(free_blocks=19, unloading=unloading)
    assert unloading is True

    unloading = watermark.update(free_blocks=20, unloading=unloading)
    assert unloading is False


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (0, 10),
        (10, 0),
        (-1, 10),
        (10, 9),
    ],
)
def test_free_block_watermark_validation(low, high):
    with pytest.raises(ValueError):
        FreeBlockWatermark(low=low, high=high)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (None, 10),
        (10, None),
        (0, 10),
        (10, 0),
        (-1, 10),
        (10, 9),
    ],
)
def test_agentic_kv_config_watermark_validation(low, high):
    with pytest.raises(ValueError):
        AgenticKVConfig(
            free_gpu_blocks_low_watermark=low,
            free_gpu_blocks_high_watermark=high,
        )


def test_agentic_kv_config_rejects_negative_cached_free_target():
    with pytest.raises(ValueError):
        AgenticKVConfig(max_cached_free_blocks=-1)
