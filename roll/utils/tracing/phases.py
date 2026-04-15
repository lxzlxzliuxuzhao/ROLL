"""
Phase definitions for tracing.

Phases are batch-level concepts used for categorizing spans
and for HTML visualization grouping.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TracePhaseDefinition:
    name: str
    description: str
    parent: str | None = None


# Updated phase definitions using unified naming
TRACE_PHASE_DEFINITIONS: tuple[TracePhaseDefinition, ...] = (
    TracePhaseDefinition("pipeline", "Top-level training step envelope."),
    TracePhaseDefinition("scheduler", "Scheduler, queue, and routing coordination phase.", parent="pipeline"),
    TracePhaseDefinition("inference", "Inference-side generation and model execution phase.", parent="pipeline"),
    TracePhaseDefinition("rollout", "Environment-side rollout and sample construction phase.", parent="pipeline"),
    TracePhaseDefinition("trajectory", "Tool-use or environment-call execution nested inside rollout.", parent="rollout"),
    TracePhaseDefinition("cache", "KV cache eviction nested inside tool-call handling.", parent="trajectory"),
    TracePhaseDefinition("reward", "Reward and reference-log-prob computation phase.", parent="pipeline"),
    TracePhaseDefinition("policy_eval", "Old policy log-prob and value recomputation phase.", parent="pipeline"),
    TracePhaseDefinition("advantage", "Advantage and return construction phase.", parent="pipeline"),
    TracePhaseDefinition("training", "Actor and critic optimization phase.", parent="pipeline"),
    TracePhaseDefinition("weight_sync", "Cross-cluster parameter synchronization phase.", parent="pipeline"),
    TracePhaseDefinition("env", "Environment interaction phase.", parent="rollout"),
)


GENERATION_SUBPHASES: tuple[str, ...] = (
    "inference.prefill",
    "inference.decode",
    "inference.overhead",
    "trajectory.tool_call",
)


DEFAULT_STEP_PHASES: tuple[str, ...] = (
    "inference",
    "reward",
    "policy_eval",
    "advantage",
    "training",
    "weight_sync",
    "scheduler",
)
