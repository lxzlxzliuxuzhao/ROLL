"""
Exporter utilities for tracing visualization.

Provides stage normalization and HTML grouping for trace spans.
"""

from roll.utils.tracing.naming import (
    STAGE_GROUP_PIPELINE,
    STAGE_GROUP_SCHEDULING,
    STAGE_GROUP_INFERENCE,
    STAGE_GROUP_INTERACTION,
    STAGE_GROUP_EVALUATION,
    STAGE_GROUP_OPTIMIZATION,
    STAGE_GROUP_DIAGNOSTICS,
)


# Stage group order for HTML visualization
STAGE_GROUP_ORDER = [
    STAGE_GROUP_PIPELINE,
    STAGE_GROUP_SCHEDULING,
    STAGE_GROUP_INFERENCE,
    STAGE_GROUP_INTERACTION,
    STAGE_GROUP_EVALUATION,
    STAGE_GROUP_OPTIMIZATION,
    STAGE_GROUP_DIAGNOSTICS,
]

# Mapping from stage_group_id to display labels
STAGE_GROUP_LABELS = {
    STAGE_GROUP_PIPELINE: "流水线",
    STAGE_GROUP_SCHEDULING: "调度",
    STAGE_GROUP_INFERENCE: "模型推理",
    STAGE_GROUP_INTERACTION: "交互",
    STAGE_GROUP_EVALUATION: "评估",
    STAGE_GROUP_OPTIMIZATION: "优化",
    STAGE_GROUP_DIAGNOSTICS: "诊断",
}


def _normalize_stage(span_name: str) -> tuple[str, str, str]:
    """
    Normalize a span name into (stage_id, stage_label, stage_group).

    Args:
        span_name: The span name in `<domain>.<operation>` format.

    Returns:
        Tuple of (stage_id, stage_label, stage_group).
    """
    if "." not in span_name:
        return span_name, span_name, STAGE_GROUP_DIAGNOSTICS

    domain, operation = span_name.split(".", 1)

    # Map domain to (stage_label, stage_group)
    domain_map = {
        "pipeline": ("Pipeline Step", STAGE_GROUP_PIPELINE),
        "scheduler": ("Scheduler", STAGE_GROUP_SCHEDULING),
        "weight_sync": ("Weight Sync", STAGE_GROUP_SCHEDULING),
        "rollout": ("Rollout", STAGE_GROUP_INTERACTION),
        "inference": ("Inference", STAGE_GROUP_INFERENCE),
        "cache": ("Cache", STAGE_GROUP_INFERENCE),
        "reward": ("Reward", STAGE_GROUP_EVALUATION),
        "policy_eval": ("Policy Eval", STAGE_GROUP_EVALUATION),
        "advantage": ("Advantage", STAGE_GROUP_OPTIMIZATION),
        "training": ("Training", STAGE_GROUP_OPTIMIZATION),
        "trajectory": ("Trajectory", STAGE_GROUP_INTERACTION),
        "env": ("Environment", STAGE_GROUP_INTERACTION),
    }

    stage_label, stage_group = domain_map.get(domain, (domain, STAGE_GROUP_DIAGNOSTICS))
    stage_id = span_name

    return stage_id, stage_label, stage_group
