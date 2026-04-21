import pytest

from roll.pipeline.agentic.akv import (
    FreeBlockWatermark,
    ResumeBoundaryKind,
    TrajectoryKVSession,
    TrajectoryKVSessionState,
    WaitReason,
)


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
