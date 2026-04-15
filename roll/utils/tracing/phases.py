"""
Phase definitions for tracing stage grouping.

Phases are batch-level concepts used for categorizing spans
and for HTML visualization grouping.
"""

from roll.utils.tracing.naming import (
    PHASE_PIPELINE,
    PHASE_SCHEDULER,
    PHASE_INFERENCE,
    PHASE_ROLLOUT,
    PHASE_TRAJECTORY,
    PHASE_CACHE,
    PHASE_REWARD,
    PHASE_POLICY_EVAL,
    PHASE_ADVANTAGE,
    PHASE_TRAINING,
    PHASE_WEIGHT_SYNC,
    PHASE_ENV,
)


# Mapping from span name prefix to phase
SPAN_TO_PHASE = {
    "pipeline": PHASE_PIPELINE,
    "scheduler": PHASE_SCHEDULER,
    "weight_sync": PHASE_WEIGHT_SYNC,
    "rollout": PHASE_ROLLOUT,
    "trajectory": PHASE_TRAJECTORY,
    "inference": PHASE_INFERENCE,
    "cache": PHASE_CACHE,
    "reward": PHASE_REWARD,
    "policy_eval": PHASE_POLICY_EVAL,
    "advantage": PHASE_ADVANTAGE,
    "training": PHASE_TRAINING,
    "env": PHASE_ENV,
}


def get_phase(span_name: str) -> str:
    """
    Get the phase for a given span name.

    Args:
        span_name: The span name in `<domain>.<operation>` format.

    Returns:
        The phase name for the span.
    """
    if "." in span_name:
        domain = span_name.split(".")[0]
        return SPAN_TO_PHASE.get(domain, domain)
    return span_name


# All valid phases
ALL_PHASES = [
    PHASE_PIPELINE,
    PHASE_SCHEDULER,
    PHASE_INFERENCE,
    PHASE_ROLLOUT,
    PHASE_TRAJECTORY,
    PHASE_CACHE,
    PHASE_REWARD,
    PHASE_POLICY_EVAL,
    PHASE_ADVANTAGE,
    PHASE_TRAINING,
    PHASE_WEIGHT_SYNC,
    PHASE_ENV,
]
