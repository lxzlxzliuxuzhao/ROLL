from dataclasses import dataclass, field
from typing import Any

from roll.pipeline.agentic.akv.session import (
    ResumeBoundaryKind,
    TrajectoryKVSession,
    WaitReason,
)


@dataclass
class AgenticKVRuntime:
    enabled: bool
    model_identity: str
    _sessions: dict[str, TrajectoryKVSession] = field(default_factory=dict)

    def start_session(self, session_id: str, traj_id: str) -> TrajectoryKVSession:
        session = TrajectoryKVSession.create(
            session_id=session_id,
            traj_id=traj_id,
            model_identity=self.model_identity,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> TrajectoryKVSession:
        return self._sessions[session_id]

    def build_request_meta(
        self,
        session_id: str,
        request_id: str,
        env_step: int,
        resume_point_id: str | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        session.mark_running(request_id=request_id)
        if resume_point_id is not None and resume_point_id not in session.resume_points:
            raise KeyError(resume_point_id)
        return {
            "enabled": True,
            "session_id": session_id,
            "latest_resume_point_id": resume_point_id or session.latest_resume_point_id,
            "candidate_resume_point_id": f"{session.session_id}:rp:{session.generation_index + 1}",
            "boundary_kind": ResumeBoundaryKind.REQUEST_END_TOOL_CALL.value,
            "wait_reason": WaitReason.TOOL_WAIT.value,
            "env_step": env_step,
            "request_configs": {"lmcache.tag.akv_session": session.session_id},
        }

    def complete_request(
        self,
        session_id: str,
        request_id: str,
        env_step: int,
        finish_reasons: list[str],
        tool_names: list[str],
        candidate_resume_point_id: str,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        if tool_names and "stop" in finish_reasons:
            resume_point = session.mark_waiting(
                request_id=request_id,
                resume_point_id=candidate_resume_point_id,
                boundary_kind=ResumeBoundaryKind.REQUEST_END_TOOL_CALL,
                wait_reason=WaitReason.TOOL_WAIT,
                env_step=env_step,
            )
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "resume_point_id": resume_point.resume_point_id,
                "wait_reason": WaitReason.TOOL_WAIT.value,
            }
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "resume_point_id": session.latest_resume_point_id,
        }

    def finish_session(self, session_id: str) -> None:
        self.get_session(session_id).mark_finished()
