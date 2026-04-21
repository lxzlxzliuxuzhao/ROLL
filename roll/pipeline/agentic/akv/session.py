from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class TrajectoryKVSessionState(str, Enum):
    RUNNING = "running"
    WAITING_EXTERNAL = "waiting_external"
    FINISHED = "finished"
    INVALIDATED = "invalidated"


class ResumeBoundaryKind(str, Enum):
    TURN_END = "turn_end"
    REQUEST_END = "request_end"
    REQUEST_END_TOOL_CALL = "request_end_tool_call"


class WaitReason(str, Enum):
    TOOL_WAIT = "tool_wait"
    ENV_WAIT = "env_wait"
    RETRY_WAIT = "retry_wait"


@dataclass(frozen=True)
class ResumePoint:
    session_id: str
    resume_point_id: str
    boundary_kind: ResumeBoundaryKind
    env_step: int
    request_id: str


@dataclass
class TrajectoryKVSession:
    session_id: str
    traj_id: str
    model_identity: str
    state: TrajectoryKVSessionState = TrajectoryKVSessionState.RUNNING
    generation_index: int = 0
    current_request_id: Optional[str] = None
    latest_resume_point_id: Optional[str] = None
    wait_reason: Optional[WaitReason] = None
    invalidation_reason: Optional[str] = None
    resume_points: Dict[str, ResumePoint] = field(default_factory=dict)

    @classmethod
    def create(cls, session_id: str, traj_id: str, model_identity: str) -> "TrajectoryKVSession":
        return cls(session_id=session_id, traj_id=traj_id, model_identity=model_identity)

    def mark_running(self, request_id: str) -> None:
        if self.state in {
            TrajectoryKVSessionState.FINISHED,
            TrajectoryKVSessionState.INVALIDATED,
        }:
            raise ValueError(f"cannot transition {self.state.value} session to running")
        self.state = TrajectoryKVSessionState.RUNNING
        self.current_request_id = request_id
        self.wait_reason = None

    def mark_waiting(
        self,
        request_id: str,
        resume_point_id: str,
        boundary_kind: ResumeBoundaryKind,
        wait_reason: WaitReason,
        env_step: int,
    ) -> ResumePoint:
        if self.state != TrajectoryKVSessionState.RUNNING:
            raise ValueError(f"cannot transition {self.state.value} session to waiting_external")
        if self.current_request_id is None:
            raise ValueError("cannot transition to waiting_external without an active request")
        if self.current_request_id != request_id:
            raise ValueError(
                f"cannot transition to waiting_external with request_id={request_id!r} "
                f"while active request_id={self.current_request_id!r}"
            )

        resume_point = ResumePoint(
            session_id=self.session_id,
            resume_point_id=resume_point_id,
            boundary_kind=boundary_kind,
            env_step=env_step,
            request_id=request_id,
        )
        self.resume_points[resume_point_id] = resume_point
        self.latest_resume_point_id = resume_point_id
        self.wait_reason = wait_reason
        self.state = TrajectoryKVSessionState.WAITING_EXTERNAL
        self.current_request_id = None
        self.generation_index += 1
        return resume_point

    def mark_finished(self) -> None:
        if self.state in {
            TrajectoryKVSessionState.FINISHED,
            TrajectoryKVSessionState.INVALIDATED,
        }:
            raise ValueError(f"cannot rewrite terminal session state {self.state.value} via mark_finished")
        self.state = TrajectoryKVSessionState.FINISHED
        self.current_request_id = None
        self.wait_reason = None

    def mark_invalidated(self, reason: str) -> None:
        if self.state in {
            TrajectoryKVSessionState.FINISHED,
            TrajectoryKVSessionState.INVALIDATED,
        }:
            raise ValueError(f"cannot rewrite terminal session state {self.state.value} via mark_invalidated")
        self.state = TrajectoryKVSessionState.INVALIDATED
        self.current_request_id = None
        self.wait_reason = None
        self.invalidation_reason = reason
