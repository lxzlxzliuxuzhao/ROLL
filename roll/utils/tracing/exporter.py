import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from statistics import median
from pathlib import Path
from typing import Any

from roll.utils.logging import get_logger


logger = get_logger()

_TIMESTAMP_CLUSTER_SLACK_NS = 7 * 24 * 60 * 60 * 1_000_000_000
_SESSION_CLUSTER_PADDING_NS = 10 * 60 * 1_000_000_000
_PREVIEW_LIMIT = 220

_STAGE_GROUP_ORDER = {
    "Pipeline": 0,
    "Scheduling": 1,
    "Inference": 2,
    "Interaction": 3,
    "Evaluation": 4,
    "Optimization": 5,
    "Diagnostics": 6,
}

# Chinese labels for display
_STAGE_GROUP_LABELS = {
    "Pipeline": "流水线",
    "Scheduling": "调度",
    "Inference": "模型推理",
    "Interaction": "交互",
    "Evaluation": "评估",
    "Optimization": "优化",
    "Diagnostics": "诊断",
}


def _slugify(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_").replace(" ", "_").lower()


def _humanize_token(value: str) -> str:
    return value.replace("_", " ").replace("/", " ").strip().title()


def _short_id(value: Any, limit: int = 8) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit]


def _clip_text(value: Any, limit: int = _PREVIEW_LIMIT) -> str:
    if value is None:
        return ""
    normalized = " ".join(str(value).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)] + "…"


def _spec(stage_id: str, label: str, group: str, description: str) -> dict[str, str]:
    return {
        "stage_id": stage_id,
        "stage_label": label,
        "stage_group": group,
        "stage_description": description,
    }


def _normalize_stage(name: str, category: str) -> dict[str, str]:
    # Pipeline step level spans
    if name == "pipeline.step":
        return _spec("step_total", "训练步骤", "Pipeline", "覆盖整个优化步骤的总时间，包含 rollout、奖励计算和参数更新。")

    # Scheduler spans
    if name == "scheduler.suspend":
        return _spec("scheduler_suspend", "暂停调度", "Scheduling", "暂停请求接入和分发，便于安全地切换调度状态。")
    if name == "scheduler.suspend_for_train":
        return _spec("scheduler_suspend_for_train", "为训练暂停推理", "Scheduling", "暂停推理，释放 colocated 资源给训练阶段使用。")
    if name == "scheduler.shrink":
        return _spec("scheduler_shrink", "收缩采样器", "Scheduling", "减少活跃推理 worker，释放 GPU 给训练。")
    if name == "scheduler.expand":
        return _spec("scheduler_expand", "扩容采样器", "Scheduling", "恢复 sampler 路由，让更多 GPU 重新参与推理。")

    # Weight sync spans
    if name == "weight_sync.offload_train":
        return _spec("weight_sync_offload_train", "卸载训练态", "Scheduling", "在 rollout 和权重同步前，先把训练侧模型状态移开。")
    if name == "weight_sync.sync_weights":
        return _spec("weight_sync_sync_weights", "同步模型权重", "Optimization", "把新权重广播到后续需要使用它们的 worker。")
    if name == "weight_sync.offload_infer":
        return _spec("weight_sync_offload_infer", "卸载推理态", "Scheduling", "异步训练期间临时卸载推理侧模型状态。")
    if name == "weight_sync.load_infer":
        return _spec("weight_sync_load_infer", "加载推理态", "Inference", "在开始服务请求前重新加载推理权重和缓存状态。")

    # Rollout spans
    if name == "rollout.get_batch":
        return _spec("rollout_get_batch", "拉取 Rollout 批次", "Scheduling", "等待 queue manager 返回一批已完成的 rollout。")
    if name == "rollout.put_batch":
        return _spec("rollout_put_batch", "回传 Rollout", "Scheduling", "把完成的 rollout 结果回推给 queue manager。")
    if name == "rollout.wait_worker":
        return _spec("rollout_wait_worker", "排队等待", "Scheduling", "请求真正开始执行前，在队列里等待 worker 的时间。")
    if name == "rollout.collect_batch":
        return _spec("rollout_collect_batch", "收集 Rollout 批次", "Pipeline", "取回一个完整 rollout batch，并附带 rollout 侧指标。")

    # Inference spans
    if name == "inference.generate":
        return _spec("inference_generate", "生成轮次", "Inference", "单轮模型生成的外层包裹 span。")
    if name == "inference.request":
        return _spec("inference_request", "LLM 请求", "Inference", "单轮请求的端到端模型推理时间。")
    if name == "inference.prefill":
        return _spec("inference_prefill", "Prefill（提示编码）", "Inference", "对 prompt 做编码并建立 KV cache，再进入生成。这个阶段通常随 prompt 长度增长。")
    if name == "inference.decode":
        return _spec("inference_decode", "Decode（逐 token 解码）", "Inference", "进入自回归逐 token 生成阶段。这个阶段通常随输出 token 数增长。")
    if name == "inference.overhead":
        return _spec("inference_overhead", "推理额外开销", "Inference", "不能直接归因到 Prefill 或 Decode 的推理额外耗时。")
    if name == "inference.metrics":
        return _spec("inference_metrics", "推理指标", "Diagnostics", "伴随生成时间一起记录的瞬时指标。")

    # Trajectory spans
    if name == "trajectory.lifetime":
        return _spec("trajectory_lifetime", "轨迹生命周期", "Interaction", "一条环境轨迹从开始到结束的总时长。")
    if name == "trajectory.step":
        return _spec("trajectory_step", "环境轮次", "Interaction", "agent 与环境交互的一轮。")
    if name == "trajectory.tool_call":
        return _spec("trajectory_tool_call", "工具执行", "Interaction", "执行模型产出的工具调用。")

    # Cache spans
    if name == "cache.eviction":
        return _spec("cache_eviction", "KV Cache 淘汰", "Inference", "KV cache 淘汰操作。")
    if name == "cache.prefetch":
        return _spec("cache_prefetch", "KV Cache 预取", "Inference", "KV cache 预取操作。")

    # Reward spans
    if name == "reward.ref_logprob":
        return _spec("reward_ref_logprob", "参考策略 LogProb", "Evaluation", "计算参考策略的 logprob，用于 KL 或奖励整形。")
    if name == "reward.compute":
        return _spec("reward_compute", "响应级奖励", "Evaluation", "在 token 级奖励展开前，先聚合响应级奖励。")
    if name == "reward.token_level":
        return _spec("reward_token_level", "Token 级奖励", "Evaluation", "把响应级奖励展开成 token 级训练目标。")

    # Policy eval spans
    if name == "policy_eval.logprob":
        return _spec("policy_eval_logprob", "旧策略 LogProb", "Evaluation", "重算或恢复旧策略 logprob，以及可选的 critic value。")

    # Advantage spans
    if name == "advantage.compute":
        return _spec("advantage_compute", "优势计算", "Evaluation", "把奖励和值函数变成最终训练使用的 advantage。")

    # Training spans
    if name == "training.actor_update":
        return _spec("training_actor_update", "策略更新", "Optimization", "执行 actor 训练；如果启用 critic，也会在这里一起训练。")

    # Environment spans
    if name == "env.reset":
        return _spec("env_reset", "环境重置", "Interaction", "重置环境到初始状态。")
    if name == "env.step":
        return _spec("env_step", "环境状态推进", "Interaction", "把模型回复作用到环境上，并拿回下一轮状态。")
    if name == "env.format_response":
        return _spec("env_format_response", "环境侧格式化回复", "Interaction", "把模型输出转换成 agent / iflow 需要的 response payload。")
    if name == "env.fetch_request":
        return _spec("env_fetch_request", "环境侧拉取下一请求", "Interaction", "把当前回复喂回 agent/sandbox，并等待下一轮 request。")
    if name == "env.check_termination":
        return _spec("env_check_termination", "终止判定", "Interaction", "检查这一轮后是否结束 episode，必要时触发收尾逻辑。")
    if name == "env.parse_request":
        return _spec("env_parse_request", "解析下一轮消息", "Interaction", "把下一轮 request payload 解析成 messages 和 tools。")
    if name == "env.reward_test":
        return _spec("env_reward_test", "收尾测试 / 奖励评估", "Interaction", "episode 结束时执行测试并计算最终奖励。")
    if name == "env.restart_session":
        return _spec("env_restart_session", "重启会话", "Interaction", "在同一 sandbox 内启动下一段 session。")
    if name == "env.close":
        return _spec("env_close", "关闭 Sandbox", "Interaction", "episode 结束后停止 sandbox 并释放资源。")
    if name == "env.start_agent":
        return _spec("env_start_agent", "启动 Agent", "Interaction", "在 sandbox 内启动 agent 进程。")
    if name == "env.fetch_init_request":
        return _spec("env_fetch_init_request", "拉取初始请求", "Interaction", "启动 agent 后获取第一轮 request。")
    if name == "env.parse_init_request":
        return _spec("env_parse_init_request", "解析初始消息", "Interaction", "把初始 request 解析成 messages 和 tools。")

    # Category fallbacks
    if category == "inference":
        return _spec("inference_misc", "推理活动", "Inference", "辅助性的推理子阶段。")
    if category == "scheduler":
        return _spec("scheduler_misc", "调度活动", "Scheduling", "辅助性的调度或队列管理阶段。")
    if category == "reward":
        return _spec("reward_misc", "奖励处理", "Evaluation", "辅助性的奖励处理阶段。")
    if category == "training":
        return _spec("training_misc", "策略更新活动", "Optimization", "辅助性的策略更新阶段。")
    if category == "weight_sync":
        return _spec("weight_sync_misc", "权重同步活动", "Optimization", "辅助性的权重同步阶段。")
    if category == "rollout":
        return _spec("rollout_misc", "Rollout 活动", "Interaction", "辅助性的 rollout 阶段。")
    if category == "trajectory":
        return _spec("trajectory_misc", "轨迹活动", "Interaction", "辅助性的轨迹阶段。")
    if category == "pipeline":
        return _spec("pipeline_misc", "流水线活动", "Pipeline", "辅助性的流水线控制阶段。")
    if category == "policy_eval":
        return _spec("policy_eval_misc", "策略评估活动", "Evaluation", "辅助性的策略评估阶段。")
    if category == "advantage":
        return _spec("advantage_misc", "优势活动", "Evaluation", "辅助性的优势计算阶段。")
    if category == "cache":
        return _spec("cache_misc", "缓存活动", "Inference", "辅助性的缓存操作阶段。")
    if category == "env":
        return _spec("env_misc", "环境活动", "Interaction", "辅助性的环境交互阶段。")

    label = _humanize_token(name.split(".")[-1] if "." in name else category)
    return _spec(_slugify(name or category), label, "Diagnostics", "尚未分类的 trace 阶段。")


def _event_title(stage: dict[str, str], attrs: dict[str, Any], raw_name: str) -> str:
    env_step = attrs.get("env_step")
    label = stage["stage_label"]
    if env_step is not None and stage["stage_id"] in {
        "env_turn",
        "llm_request",
        "prompt_encoding",
        "token_decoding",
        "generation_overhead",
        "queue_wait",
        "tool_execution",
        "environment_transition",
    }:
        return f"{label} · 第{int(env_step) + 1}轮"
    if stage["stage_id"] == "trajectory_lifetime":
        return "轨迹生命周期"
    if stage["stage_id"] == "step_total":
        return "完整训练步骤"
    return label or raw_name


def _event_style(stage_id: str, raw_name: str) -> str:
    if stage_id in {"step_total", "trajectory_lifetime"}:
        return "background"
    if stage_id in {"env_turn", "generation_turn"} or raw_name == "generation":
        return "muted"
    return "standard"


def _row_preview(attrs: dict[str, Any]) -> str:
    return _clip_text(attrs.get("last_message_preview") or attrs.get("response_preview") or "")


def _env_step_from_sample_id(sample_id: Any) -> int | None:
    if not sample_id:
        return None
    text = str(sample_id)
    marker = ":step:"
    if marker not in text:
        return None
    try:
        return int(text.rsplit(marker, 1)[-1])
    except ValueError:
        return None


def _build_raw_rows(spans: list[dict[str, Any]], start_ref: int) -> list[dict[str, Any]]:
    by_id = {span["span_id"]: span for span in spans}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for span in spans:
        parent_id = span.get("parent_id")
        if parent_id and parent_id in by_id:
            children[parent_id].append(span["span_id"])
        else:
            roots.append(span["span_id"])

    rows: list[dict[str, Any]] = []

    def visit(span_id: str, depth: int) -> None:
        span = by_id[span_id]
        attrs = span.get("attrs", {}) or {}
        stage = _normalize_stage(span["name"], span["category"])
        row = {
            "span_id": span["span_id"],
            "parent_id": span.get("parent_id"),
            "name": span["name"],
            "phase": span["phase"],
            "category": span["category"],
            "depth": depth,
            "step": span.get("step"),
            "sample_id": span.get("sample_id"),
            "traj_id": span.get("traj_id"),
            "process_label": span.get("process_label"),
            "duration_ms": span["duration_ms"],
            "start_offset_ms": round((span["start_time_ns"] - start_ref) / 1_000_000, 6),
            "end_offset_ms": round((span["end_time_ns"] - start_ref) / 1_000_000, 6),
            "attrs": attrs,
            "has_children": bool(children.get(span_id)),
            "stage_id": stage["stage_id"],
            "stage_label": stage["stage_label"],
            "stage_group": stage["stage_group"],
            "stage_description": stage["stage_description"],
            "title": _event_title(stage, attrs, span["name"]),
            "event_style": _event_style(stage["stage_id"], span["name"]),
            "preview": _row_preview(attrs),
            "env_step": attrs.get("env_step", _env_step_from_sample_id(span.get("sample_id"))),
            "request_id": attrs.get("request_id"),
        }
        rows.append(row)
        for child_id in sorted(children.get(span_id, []), key=lambda item: by_id[item]["start_time_ns"]):
            visit(child_id, depth + 1)

    for root_id in sorted(roots, key=lambda item: by_id[item]["start_time_ns"]):
        visit(root_id, 0)
    return rows


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator in (0, 0.0, None):
        return None
    return numerator / denominator


def _mode_label(mode: Any) -> str:
    return {
        "train": "训练",
        "val": "验证",
        "eval": "评测",
    }.get(str(mode), str(mode))


def _enrich_inference_rows(rows: list[dict[str, Any]]) -> None:
    request_map: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in rows:
        if row["stage_id"] == "llm_request":
            request_map[(row.get("traj_id"), row.get("sample_id"))] = row

    for row in rows:
        key = (row.get("traj_id"), row.get("sample_id"))
        request_row = request_map.get(key)
        if not request_row:
            continue
        request_attrs = request_row.get("attrs", {})
        if row.get("request_id") is None:
            row["request_id"] = request_row.get("request_id") or request_attrs.get("request_id")
        if not row.get("preview"):
            row["preview"] = request_row.get("preview", "")
        row["request_prompt_tokens"] = request_attrs.get("prompt_tokens")
        row["request_output_tokens"] = request_attrs.get("output_tokens")
        row["request_duration_ms"] = request_row["duration_ms"]
        if row.get("env_step") is None:
            row["env_step"] = request_row.get("env_step")
        if row.get("env_step") is not None and row["stage_id"] in {
            "prompt_encoding",
            "token_decoding",
            "generation_overhead",
            "queue_wait",
        } and "Turn" not in row["title"]:
            row["title"] = f"{row['stage_label']} · Turn {int(row['env_step']) + 1}"

        prompt_tokens = row.get("request_prompt_tokens")
        output_tokens = row.get("request_output_tokens")
        if row["stage_id"] == "prompt_encoding" and prompt_tokens:
            value = _safe_ratio(row["duration_ms"] * 1000.0, float(prompt_tokens))
            if value is not None:
                row["performance_label"] = "Prefill 单位成本"
                row["performance_value"] = round(value, 4)
                row["performance_unit"] = "ms / 1k tok"
        elif row["stage_id"] == "token_decoding" and output_tokens:
            value = _safe_ratio(row["duration_ms"], float(output_tokens))
            if value is not None:
                row["performance_label"] = "Decode 单位成本"
                row["performance_value"] = round(value, 6)
                row["performance_unit"] = "ms / tok"
        elif row["stage_id"] == "queue_wait":
            total = request_row["duration_ms"]
            value = _safe_ratio(row["duration_ms"] * 100.0, total)
            if value is not None:
                row["performance_label"] = "排队占比"
                row["performance_value"] = round(value, 2)
                row["performance_unit"] = "%"
        elif row["stage_id"] == "generation_overhead":
            total = request_row["duration_ms"]
            value = _safe_ratio(row["duration_ms"] * 100.0, total)
            if value is not None:
                row["performance_label"] = "额外开销占比"
                row["performance_value"] = round(value, 2)
                row["performance_unit"] = "%"


def _build_inference_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    request_rows = [row for row in rows if row["stage_id"] == "llm_request"]
    sample_groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("traj_id") and row.get("sample_id"):
            sample_groups[(row.get("traj_id"), row.get("sample_id"))].append(row)

    requests = []
    total_prefill_ms = 0.0
    total_decode_ms = 0.0
    total_queue_ms = 0.0
    total_overhead_ms = 0.0
    total_prompt_tokens = 0
    total_output_tokens = 0

    for key, group_rows in sample_groups.items():
        request_row = next((row for row in group_rows if row["stage_id"] == "llm_request"), None)
        if request_row is None:
            continue
        prompt_tokens = int(request_row.get("request_prompt_tokens") or request_row["attrs"].get("prompt_tokens") or 0)
        output_tokens = int(request_row.get("request_output_tokens") or request_row["attrs"].get("output_tokens") or 0)
        prefill_ms = sum(row["duration_ms"] for row in group_rows if row["stage_id"] == "prompt_encoding")
        decode_ms = sum(row["duration_ms"] for row in group_rows if row["stage_id"] == "token_decoding")
        queue_ms = sum(row["duration_ms"] for row in group_rows if row["stage_id"] == "queue_wait")
        overhead_ms = sum(row["duration_ms"] for row in group_rows if row["stage_id"] == "generation_overhead")
        total_ms = request_row["duration_ms"]

        total_prefill_ms += prefill_ms
        total_decode_ms += decode_ms
        total_queue_ms += queue_ms
        total_overhead_ms += overhead_ms
        total_prompt_tokens += prompt_tokens
        total_output_tokens += output_tokens

        requests.append(
            {
                "traj_id": request_row.get("traj_id"),
                "sample_id": request_row.get("sample_id"),
                "request_id": request_row.get("request_id"),
                "env_step": request_row.get("env_step"),
                "mode": request_row["attrs"].get("mode"),
                "tag": request_row["attrs"].get("tag"),
                "env_id": request_row["attrs"].get("env_id"),
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_ms": round(total_ms, 6),
                "queue_ms": round(queue_ms, 6),
                "prefill_ms": round(prefill_ms, 6),
                "decode_ms": round(decode_ms, 6),
                "overhead_ms": round(overhead_ms, 6),
                "prefill_ms_per_1k_prompt_tokens": round(prefill_ms * 1000.0 / prompt_tokens, 4) if prompt_tokens else None,
                "decode_ms_per_token": round(decode_ms / output_tokens, 6) if output_tokens else None,
                "preview": request_row.get("preview", ""),
            }
        )

    requests.sort(key=lambda item: item["total_ms"], reverse=True)
    inference_total_ms = total_prefill_ms + total_decode_ms + total_queue_ms + total_overhead_ms
    return {
        "request_count": len(requests),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_prefill_ms": round(total_prefill_ms, 6),
        "total_decode_ms": round(total_decode_ms, 6),
        "total_queue_ms": round(total_queue_ms, 6),
        "total_overhead_ms": round(total_overhead_ms, 6),
        "avg_prompt_tokens": round(total_prompt_tokens / len(requests), 2) if requests else 0.0,
        "avg_output_tokens": round(total_output_tokens / len(requests), 2) if requests else 0.0,
        "avg_prefill_ms": round(total_prefill_ms / len(requests), 6) if requests else 0.0,
        "avg_decode_ms": round(total_decode_ms / len(requests), 6) if requests else 0.0,
        "prefill_ms_per_1k_prompt_tokens": round(total_prefill_ms * 1000.0 / total_prompt_tokens, 4) if total_prompt_tokens else None,
        "decode_ms_per_token": round(total_decode_ms / total_output_tokens, 6) if total_output_tokens else None,
        "prefill_share_pct": round(total_prefill_ms * 100.0 / inference_total_ms, 2) if inference_total_ms else 0.0,
        "decode_share_pct": round(total_decode_ms * 100.0 / inference_total_ms, 2) if inference_total_ms else 0.0,
        "queue_share_pct": round(total_queue_ms * 100.0 / inference_total_ms, 2) if inference_total_ms else 0.0,
        "overhead_share_pct": round(total_overhead_ms * 100.0 / inference_total_ms, 2) if inference_total_ms else 0.0,
        "slow_requests": requests[:8],
    }


def _build_stage_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_totals: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = stage_totals.setdefault(
            row["stage_id"],
            {
                "stage_id": row["stage_id"],
                "stage_label": row["stage_label"],
                "stage_group": row["stage_group"],
                "stage_description": row["stage_description"],
                "span_count": 0,
                "total_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "raw_names": set(),
            },
        )
        bucket["span_count"] += 1
        bucket["total_duration_ms"] += row["duration_ms"]
        bucket["max_duration_ms"] = max(bucket["max_duration_ms"], row["duration_ms"])
        bucket["raw_names"].add(row["name"])

    items = []
    for item in stage_totals.values():
        raw_names = sorted(item["raw_names"])
        items.append(
            {
                "stage_id": item["stage_id"],
                "stage_label": item["stage_label"],
                "stage_group": item["stage_group"],
                "stage_description": item["stage_description"],
                "span_count": item["span_count"],
                "total_duration_ms": round(item["total_duration_ms"], 6),
                "max_duration_ms": round(item["max_duration_ms"], 6),
                "raw_names": raw_names,
                "raw_name_preview": raw_names[:3],
            }
        )
    return sorted(
        items,
        key=lambda item: (
            _STAGE_GROUP_ORDER.get(item["stage_group"], 99),
            -item["total_duration_ms"],
            item["stage_label"],
        ),
    )


def _build_trajectory_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("traj_id"):
            grouped[row["traj_id"]].append(row)

    items = []
    for traj_id, traj_rows in grouped.items():
        traj_rows.sort(key=lambda row: (row["start_offset_ms"], row["end_offset_ms"]))
        attrs_source = next((row["attrs"] for row in traj_rows if row["stage_id"] == "trajectory_lifetime"), traj_rows[0]["attrs"])
        env_steps = sorted(
            {
                int(row["env_step"])
                for row in traj_rows
                if row.get("env_step") is not None
            }
        )
        request_rows = [row for row in traj_rows if row["stage_id"] == "llm_request"]
        tool_rows = [row for row in traj_rows if row["stage_id"] == "tool_execution"]
        stage_counts = Counter(row["stage_label"] for row in traj_rows if row["stage_id"] != "trajectory_lifetime")
        preview = next((row["preview"] for row in request_rows if row["preview"]), "")
        if not preview:
            preview = next((row["preview"] for row in tool_rows if row["preview"]), "")
        items.append(
            {
                "traj_id": traj_id,
                "traj_short": _short_id(traj_id, 12),
                "mode": attrs_source.get("mode", "unknown"),
                "tag": attrs_source.get("tag", "trajectory"),
                "result": attrs_source.get("result", "unknown"),
                "stop_reason": attrs_source.get("stop_reason", ""),
                "env_id": attrs_source.get("env_id"),
                "group_id": attrs_source.get("group_id"),
                "episode_id": attrs_source.get("episode_id"),
                "span_count": len(traj_rows),
                "duration_ms": round(max(row["end_offset_ms"] for row in traj_rows) - min(row["start_offset_ms"] for row in traj_rows), 6),
                "start_offset_ms": min(row["start_offset_ms"] for row in traj_rows),
                "end_offset_ms": max(row["end_offset_ms"] for row in traj_rows),
                "processes": sorted({row.get("process_label") or "process" for row in traj_rows}),
                "turn_count": len(env_steps),
                "env_steps": env_steps,
                "request_count": len(request_rows),
                "tool_call_count": len(tool_rows),
                "queue_wait_count": sum(1 for row in traj_rows if row["stage_id"] == "queue_wait"),
                "prompt_tokens_total": int(sum((row["attrs"].get("prompt_tokens") or 0) for row in request_rows)),
                "output_tokens_total": int(sum((row["attrs"].get("output_tokens") or 0) for row in request_rows)),
                "top_stages": stage_counts.most_common(4),
                "preview": preview,
            }
        )

    return sorted(
        items,
        key=lambda item: (
            item["mode"],
            item["tag"],
            item["env_id"] if item["env_id"] is not None else -1,
            item["start_offset_ms"],
        ),
    )


def _lane_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row["span_id"],
        "span_id": row["span_id"],
        "title": row["title"],
        "stage_id": row["stage_id"],
        "stage_label": row["stage_label"],
        "stage_group": row["stage_group"],
        "stage_description": row["stage_description"],
        "event_style": row["event_style"],
        "start_offset_ms": row["start_offset_ms"],
        "end_offset_ms": row["end_offset_ms"],
        "duration_ms": row["duration_ms"],
        "raw_name": row["name"],
        "raw_category": row["category"],
        "phase": row["phase"],
        "process_label": row.get("process_label"),
        "traj_id": row.get("traj_id"),
        "sample_id": row.get("sample_id"),
        "env_step": row.get("env_step"),
        "preview": row.get("preview", ""),
        "request_id": row.get("request_id"),
        "request_prompt_tokens": row.get("request_prompt_tokens"),
        "request_output_tokens": row.get("request_output_tokens"),
        "request_duration_ms": row.get("request_duration_ms"),
        "performance_label": row.get("performance_label"),
        "performance_value": row.get("performance_value"),
        "performance_unit": row.get("performance_unit"),
        "attrs": row.get("attrs", {}),
    }


def _include_in_trajectory_lane(row: dict[str, Any]) -> bool:
    return row["name"] in {
        "trajectory.lifetime",
        "trajectory.step",
        "inference.request",
        "inference.prefill",
        "inference.decode",
        "inference.overhead",
        "rollout.wait_worker",
        "trajectory.tool_call",
        "env.step",
    }


def _build_lanes(rows: list[dict[str, Any]], trajectory_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []

    step_events = [_lane_event(row) for row in rows if row.get("traj_id") is None]
    if step_events:
        lanes.append(
            {
                "lane_id": "step_overview",
                "lane_type": "step",
                "lane_label": "步骤总览",
                "lane_meta": f"{len(step_events)} 个流程控制 span",
                "sort_key": 0,
                "events": step_events,
            }
        )

    row_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("traj_id"):
            row_groups[row["traj_id"]].append(row)

    for index, traj in enumerate(trajectory_stats, start=1):
        traj_rows = [row for row in row_groups.get(traj["traj_id"], []) if _include_in_trajectory_lane(row)]
        if not traj_rows:
            continue
        traj_rows.sort(key=lambda row: (row["start_offset_ms"], row["end_offset_ms"]))
        lane_label = f"{_mode_label(traj['mode'])} · {traj['tag']} · env {traj['env_id']}"
        lane_meta = (
            f"{traj['result']}"
            + (f" / {traj['stop_reason']}" if traj["stop_reason"] else "")
            + f" · 轮次 {traj['turn_count']} · 请求 {traj['request_count']} · 工具 {traj['tool_call_count']}"
        )
        lanes.append(
            {
                "lane_id": traj["traj_id"],
                "lane_type": "trajectory",
                "lane_label": lane_label,
                "lane_meta": lane_meta,
                "sort_key": index,
                "traj_id": traj["traj_id"],
                "events": [_lane_event(row) for row in traj_rows],
            }
        )

    return lanes


def _build_overview(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    trajectory_stats: list[dict[str, Any]],
    stage_stats: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
) -> dict[str, Any]:
    request_rows = [row for row in rows if row["stage_id"] == "llm_request"]
    tool_rows = [row for row in rows if row["stage_id"] == "tool_execution"]
    preview_available = any(row.get("preview") for row in rows)
    longest_traj_ms = max((item["duration_ms"] for item in trajectory_stats), default=0.0)
    top_stage_candidates = [
        item
        for item in stage_stats
        if item["stage_id"] not in {"trajectory_lifetime", "step_total"}
    ]
    top_stage = max(top_stage_candidates or stage_stats, key=lambda item: item["total_duration_ms"], default=None)
    return {
        "total_spans": summary["total_spans"],
        "window_ms": summary["window_ms"],
        "process_count": len(summary["processes"]),
        "trajectory_count": len(trajectory_stats),
        "lane_count": len(lanes),
        "llm_request_count": len(request_rows),
        "tool_call_count": len(tool_rows),
        "prompt_tokens_total": int(sum((row["attrs"].get("prompt_tokens") or 0) for row in request_rows)),
        "output_tokens_total": int(sum((row["attrs"].get("output_tokens") or 0) for row in request_rows)),
        "longest_trajectory_ms": round(longest_traj_ms, 6),
        "preview_available": preview_available,
        "top_stage_label": top_stage["stage_label"] if top_stage else "",
        "top_stage_duration_ms": top_stage["total_duration_ms"] if top_stage else 0.0,
    }


def _load_step_spans(trace_dir: str, step: int) -> list[dict[str, Any]]:
    step_dir = Path(trace_dir) / "raw" / "steps" / f"step_{step:06d}"
    spans: list[dict[str, Any]] = []
    if not step_dir.exists():
        return spans
    for path in sorted(step_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                spans.append(json.loads(line))
    return spans


def _filter_incoherent_spans(spans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not spans:
        return spans, []

    endpoints = [
        int(timestamp)
        for span in spans
        for timestamp in (span.get("start_time_ns"), span.get("end_time_ns"))
        if timestamp is not None
    ]
    if not endpoints:
        return spans, []

    reference_ts = int(median(endpoints))
    filtered: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for span in spans:
        start_ns = int(span["start_time_ns"])
        end_ns = int(span["end_time_ns"])
        if end_ns < start_ns:
            dropped.append(span)
            continue
        if abs(start_ns - reference_ts) > _TIMESTAMP_CLUSTER_SLACK_NS:
            dropped.append(span)
            continue
        if abs(end_ns - reference_ts) > _TIMESTAMP_CLUSTER_SLACK_NS:
            dropped.append(span)
            continue
        filtered.append(span)

    return filtered, dropped


def _select_latest_session_spans(spans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root_spans = [
        span
        for span in spans
        if span.get("name") == "pipeline/step" and span.get("process_label") == "driver"
    ]
    if len(root_spans) <= 1:
        return spans, []

    latest_root = max(root_spans, key=lambda span: int(span["end_time_ns"]))
    window_start_ns = int(latest_root["start_time_ns"]) - _SESSION_CLUSTER_PADDING_NS
    window_end_ns = int(latest_root["end_time_ns"]) + _SESSION_CLUSTER_PADDING_NS

    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for span in spans:
        start_ns = int(span["start_time_ns"])
        end_ns = int(span["end_time_ns"])
        if end_ns >= window_start_ns and start_ns <= window_end_ns:
            selected.append(span)
        else:
            dropped.append(span)

    return selected, dropped


def _build_step_bundle(spans: list[dict[str, Any]], step: int) -> dict[str, Any]:
    spans, dropped_spans = _filter_incoherent_spans(spans)
    spans, stale_session_spans = _select_latest_session_spans(spans)
    dropped_spans.extend(stale_session_spans)
    if dropped_spans:
        logger.warning(
            "Dropped %s trace spans while exporting step %s. Categories=%s",
            len(dropped_spans),
            step,
            dict(Counter(span["category"] for span in dropped_spans)),
        )

    if not spans:
        return {
            "step": step,
            "summary": {
                "total_spans": 0,
                "processes": [],
                "categories": {},
                "window_ms": 0.0,
                "dropped_spans": len(dropped_spans),
                "dropped_categories": dict(Counter(span["category"] for span in dropped_spans)),
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
            "overview": {
                "total_spans": 0,
                "window_ms": 0.0,
                "process_count": 0,
                "trajectory_count": 0,
                "lane_count": 0,
                "llm_request_count": 0,
                "tool_call_count": 0,
                "prompt_tokens_total": 0,
                "output_tokens_total": 0,
                "longest_trajectory_ms": 0.0,
                "preview_available": False,
                "top_stage_label": "",
                "top_stage_duration_ms": 0.0,
            },
            "inference_summary": {
                "request_count": 0,
                "total_prompt_tokens": 0,
                "total_output_tokens": 0,
                "total_prefill_ms": 0.0,
                "total_decode_ms": 0.0,
                "total_queue_ms": 0.0,
                "total_overhead_ms": 0.0,
                "avg_prompt_tokens": 0.0,
                "avg_output_tokens": 0.0,
                "avg_prefill_ms": 0.0,
                "avg_decode_ms": 0.0,
                "prefill_ms_per_1k_prompt_tokens": None,
                "decode_ms_per_token": None,
                "prefill_share_pct": 0.0,
                "decode_share_pct": 0.0,
                "queue_share_pct": 0.0,
                "overhead_share_pct": 0.0,
                "slow_requests": [],
            },
            "stage_stats": [],
            "lanes": [],
            "rows": [],
            "trajectory_stats": [],
            "notes": [],
            "message": (
                f"No coherent traced spans were exported for step {step}."
                if dropped_spans
                else f"No traced spans were flushed for step {step}."
            ),
        }

    spans.sort(key=lambda item: (item["start_time_ns"], item["end_time_ns"]))
    start_ref = min(span["start_time_ns"] for span in spans)
    end_ref = max(span["end_time_ns"] for span in spans)
    rows = _build_raw_rows(spans, start_ref)
    _enrich_inference_rows(rows)

    category_counter = Counter(span["category"] for span in spans)
    process_labels = sorted({span.get("process_label") or "process" for span in spans})
    stage_stats = _build_stage_stats(rows)
    trajectory_stats = _build_trajectory_stats(rows)
    lanes = _build_lanes(rows, trajectory_stats)
    inference_summary = _build_inference_summary(rows)
    summary = {
        "total_spans": len(spans),
        "processes": process_labels,
        "categories": dict(category_counter),
        "window_ms": round((end_ref - start_ref) / 1_000_000, 6),
        "dropped_spans": len(dropped_spans),
        "dropped_categories": dict(Counter(span["category"] for span in dropped_spans)),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    overview = _build_overview(summary, rows, trajectory_stats, stage_stats, lanes)
    notes = ["阶段耗时是 inclusive traced time，因此不同阶段之间可能互相重叠。"]
    if inference_summary["request_count"] > 0:
        notes.append(
            "Prefill 主要受 prompt 长度和 KV cache 建立影响；Decode 主要受生成 token 数影响。"
        )
    if any(
        row["stage_id"] in {"prompt_encoding", "token_decoding"} and row.get("attrs", {}).get("source") == "fallback_local_timing"
        for row in rows
    ):
        notes.append(
            "这批已导出的 trace 中，部分 Prefill/Decode 仍然使用了较粗粒度的本地 fallback 拆分。"
            "使用新版 vLLM 子阶段采集逻辑重新跑出的 trace，会得到更可信的 Prefill / Decode 划分。"
        )
    if dropped_spans:
        notes.append(f"导出这个 step 前，已过滤掉 {len(dropped_spans)} 个时间戳异常的 span。")
    if not overview["preview_available"] and overview["llm_request_count"] > 0:
        notes.append(
            "当前这批 raw trace 还没有消息预览字段。"
            "等你用新版采集逻辑重新跑之后，页面里会直接显示输入/输出摘要。"
        )

    return {
        "step": step,
        "summary": summary,
        "overview": overview,
        "inference_summary": inference_summary,
        "stage_stats": stage_stats,
        "lanes": lanes,
        "rows": rows,
        "trajectory_stats": trajectory_stats,
        "notes": notes,
        "message": notes[1] if len(notes) > 1 else "",
    }


def _render_step_html(bundle: dict[str, Any]) -> str:
    data_json = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ROLL Trace 时间线 | step __STEP__</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 253, 248, 0.94);
      --panel-strong: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --muted-2: #8b95a5;
      --border: #d1c7b7;
      --border-soft: #e6ddcf;
      --shadow: 0 18px 50px rgba(36, 27, 18, 0.10);
      --accent: #9a3412;
      --timeline-bg: rgba(248, 244, 236, 0.96);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(194, 65, 12, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f2e9 0%, #efe7da 100%);
    }
    .page {
      padding: 20px;
      display: grid;
      gap: 16px;
    }
    .hero, .panel {
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 16px 18px;
      position: sticky;
      top: 0;
      z-index: 5;
      backdrop-filter: blur(8px);
    }
    .hero-top {
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .title {
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.02em;
    }
    .subtitle {
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .controls label {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }
    .controls input[type="range"] {
      width: 180px;
      accent-color: #92400e;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .card {
      border: 1px solid var(--border-soft);
      border-radius: 14px;
      background: rgba(255, 252, 246, 0.92);
      padding: 12px 14px;
      min-height: 96px;
    }
    .card-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .card-value {
      margin-top: 6px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .card-meta {
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }
    .legend {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px dashed rgba(156, 134, 108, 0.35);
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    .legend-item input {
      margin: 0;
      accent-color: #92400e;
    }
    .legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(0, 0, 0, 0.15);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(340px, 0.95fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      padding: 16px 18px;
    }
    .panel h3 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    .panel-note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .timeline-panel {
      padding: 0;
      overflow: hidden;
    }
    .viewport {
      background: var(--timeline-bg);
      overflow: auto;
      border-radius: 18px;
    }
    svg {
      display: block;
      min-width: 100%;
    }
    .axis-label {
      fill: var(--muted);
      font-size: 11px;
    }
    .lane-title {
      fill: var(--ink);
      font-size: 12px;
      font-weight: 600;
    }
    .lane-meta {
      fill: var(--muted);
      font-size: 11px;
    }
    .event-label {
      fill: rgba(255, 255, 255, 0.96);
      font-size: 10px;
      pointer-events: none;
    }
    .badge-row, .detail-badges {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--border-soft);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      color: var(--ink);
      background: rgba(247, 243, 235, 0.9);
    }
    .detail-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin: 0;
    }
    .detail-subtitle {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .detail-preview {
      margin-top: 12px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border-soft);
      background: #f7f2ea;
      font-size: 13px;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .detail-attrs {
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }
    .detail-attr {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 10px;
      font-size: 12px;
      line-height: 1.5;
      padding-bottom: 8px;
      border-bottom: 1px dashed rgba(209, 199, 183, 0.65);
    }
    .detail-attr:last-child {
      border-bottom: none;
      padding-bottom: 0;
    }
    .detail-key {
      color: var(--muted);
      font-weight: 600;
    }
    details {
      margin-top: 12px;
    }
    pre {
      margin: 10px 0 0;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border-soft);
      background: #f7f2ea;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 360px;
      overflow: auto;
    }
    .glossary {
      display: grid;
      gap: 10px;
    }
    .glossary-item {
      border: 1px solid var(--border-soft);
      border-radius: 12px;
      background: rgba(247, 243, 235, 0.85);
      padding: 11px 12px;
    }
    .glossary-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
    }
    .glossary-title {
      font-size: 13px;
      font-weight: 700;
    }
    .glossary-meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .trajectory-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }
    .trajectory-card {
      border: 1px solid var(--border-soft);
      border-radius: 14px;
      background: rgba(255, 252, 246, 0.92);
      padding: 13px 14px;
      cursor: pointer;
    }
    .trajectory-card:hover {
      border-color: #c9b79b;
      background: #fff8ee;
    }
    .trajectory-title {
      font-size: 14px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .trajectory-meta {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .trajectory-preview {
      margin-top: 8px;
      font-size: 12px;
      line-height: 1.55;
      color: var(--ink);
      white-space: pre-wrap;
    }
    .notes {
      display: grid;
      gap: 8px;
    }
    .note {
      border-left: 3px solid #c2410c;
      padding-left: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .tooltip {
      position: fixed;
      z-index: 8;
      pointer-events: none;
      opacity: 0;
      transition: opacity 120ms ease;
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(17, 24, 39, 0.95);
      color: #fff;
      font-size: 12px;
      line-height: 1.45;
      max-width: 360px;
      white-space: pre-wrap;
      box-shadow: 0 12px 32px rgba(17, 24, 39, 0.28);
    }
    @media (max-width: 1180px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1 class="title">ROLL Trace 时间线 | step __STEP__</h1>
          <div class="subtitle" id="subtitle"></div>
        </div>
        <div class="controls">
          <label id="zoom-label">缩放
            <input id="zoom" type="range" min="0.4" max="90" step="0.2" value="18">
          </label>
        </div>
      </div>
      <div class="cards" id="cards"></div>
      <div class="legend" id="legend"></div>
    </section>

    <div class="layout">
      <section class="panel timeline-panel">
        <div class="viewport">
          <svg id="chart"></svg>
        </div>
      </section>
      <div class="sidebar">
        <section class="panel">
          <h3>选中事件</h3>
          <div id="detail-empty" class="panel-note">点击时间线中的事件，查看它的含义、关联元数据和原始 trace 载荷。</div>
          <div id="detail-content" hidden>
            <div class="detail-title" id="detail-title"></div>
            <div class="detail-subtitle" id="detail-subtitle"></div>
            <div class="panel-note" id="detail-description"></div>
            <div class="detail-badges" id="detail-badges" style="margin-top:12px;"></div>
            <div id="detail-preview" class="detail-preview" hidden></div>
            <div class="detail-attrs" id="detail-attrs"></div>
            <details>
              <summary>原始事件 JSON</summary>
              <pre id="detail-json">{}</pre>
            </details>
          </div>
        </section>

        <section class="panel">
          <h3>阶段说明</h3>
          <div class="panel-note" style="margin-bottom:10px;">这里展示规范化后的阶段名称和解释。阶段耗时是 inclusive 的，所以不同阶段可能彼此重叠。</div>
          <div id="glossary" class="glossary"></div>
        </section>
      </div>
    </div>

    <section class="panel">
      <h3>推理拆解</h3>
      <div class="panel-note" style="margin-bottom:12px;">Prefill 是首个 token 返回前的提示编码与 KV cache 建立成本。Decode 是首 token 之后的逐 token 自回归生成成本，通常和输出长度强相关。</div>
      <div id="inference-cards" class="cards"></div>
      <div id="request-grid" class="trajectory-grid" style="margin-top:12px;"></div>
    </section>

    <section class="panel">
      <h3>轨迹摘要</h3>
      <div class="panel-note" style="margin-bottom:12px;">每张卡片对应一条 trajectory，并可以回跳到时间线里的对应 lane。</div>
      <div id="trajectory-grid" class="trajectory-grid"></div>
    </section>

    <section class="panel">
      <h3>导出说明</h3>
      <div id="notes" class="notes"></div>
    </section>
  </div>

  <div id="tooltip" class="tooltip"></div>

  <script>
    const data = __DATA_JSON__;
    const svgNS = 'http://www.w3.org/2000/svg';
    const stageColors = {
      step_total: '#8c2d04',
      rollout_collect_batch: '#1d4ed8',
      rollout_get_batch: '#78716c',
      weight_sync_offload_train: '#57534e',
      scheduler_suspend: '#6b7280',
      scheduler_suspend_for_train: '#6b7280',
      scheduler_expand: '#4b5563',
      scheduler_shrink: '#4b5563',
      weight_sync_sync_weights: '#991b1b',
      training_actor_update: '#b91c1c',
      reward_ref_logprob: '#a16207',
      policy_eval_logprob: '#6d28d9',
      reward_compute: '#b45309',
      reward_token_level: '#c2410c',
      advantage_compute: '#92400e',
      trajectory_lifetime: '#166534',
      trajectory_step: '#0f766e',
      env_step: '#0f766e',
      trajectory_tool_call: '#ea580c',
      inference_request: '#2563eb',
      inference_prefill: '#1d4ed8',
      inference_decode: '#16a34a',
      inference_overhead: '#64748b',
      rollout_wait_worker: '#7c6f64',
      weight_sync_load_infer: '#0369a1',
      weight_sync_offload_infer: '#155e75',
      rollout_put_batch: '#7c6f64',
      inference_metrics: '#475569',
    };
    const groupColors = {
      'Pipeline': '#9a3412',
      'Scheduling': '#78716c',
      'Inference': '#1d4ed8',
      'Interaction': '#0f766e',
      'Evaluation': '#b45309',
      'Optimization': '#991b1b',
      'Diagnostics': '#475569',
    };
    const timeUnits = [
      { label: 'ms', factor: 1 },
      { label: 's', factor: 1000 },
      { label: 'min', factor: 60 * 1000 },
      { label: 'h', factor: 60 * 60 * 1000 },
      { label: 'd', factor: 24 * 60 * 60 * 1000 },
      { label: 'wk', factor: 7 * 24 * 60 * 60 * 1000 },
      { label: 'mo', factor: 30.4375 * 24 * 60 * 60 * 1000 },
      { label: 'yr', factor: 365.25 * 24 * 60 * 60 * 1000 },
    ];

    const subtitle = document.getElementById('subtitle');
    const cards = document.getElementById('cards');
    const legend = document.getElementById('legend');
    const chart = document.getElementById('chart');
    const zoom = document.getElementById('zoom');
    const zoomLabel = document.getElementById('zoom-label');
    const glossary = document.getElementById('glossary');
    const inferenceCards = document.getElementById('inference-cards');
    const requestGrid = document.getElementById('request-grid');
    const trajectoryGrid = document.getElementById('trajectory-grid');
    const notes = document.getElementById('notes');
    const tooltip = document.getElementById('tooltip');

    const detailEmpty = document.getElementById('detail-empty');
    const detailContent = document.getElementById('detail-content');
    const detailTitle = document.getElementById('detail-title');
    const detailSubtitle = document.getElementById('detail-subtitle');
    const detailDescription = document.getElementById('detail-description');
    const detailBadges = document.getElementById('detail-badges');
    const detailPreview = document.getElementById('detail-preview');
    const detailAttrs = document.getElementById('detail-attrs');
    const detailJson = document.getElementById('detail-json');

    const stageStats = data.stage_stats || [];
    const lanes = data.lanes || [];
    const trajectoryStats = data.trajectory_stats || [];
    const overview = data.overview || {};
    const inferenceSummary = data.inference_summary || {};
    const stageMap = new Map(stageStats.map((item) => [item.stage_id, item]));
    const trajMap = new Map(trajectoryStats.map((item) => [item.traj_id, item]));
    const eventMap = new Map();
    lanes.forEach((lane) => lane.events.forEach((event) => eventMap.set(event.event_id, { ...event, lane_id: lane.lane_id, lane_label: lane.lane_label })));

    const activeStages = new Set(stageStats.map((item) => item.stage_id));
    const windowMs = Math.max(data.summary?.window_ms || 1, 1);
    const timeUnit = chooseTimeUnit(windowMs);
    const windowUnits = Math.max(windowMs / timeUnit.factor, 1);
    let selectedEventId = null;

    function createSvg(tag) {
      return document.createElementNS(svgNS, tag);
    }

    function chooseTimeUnit(durationMs) {
      for (const unit of timeUnits) {
        if (durationMs / unit.factor <= 500) {
          return unit;
        }
      }
      return timeUnits[timeUnits.length - 1];
    }

    function resolveColor(event) {
      return stageColors[event.stage_id] || groupColors[event.stage_group] || '#475569';
    }

    function formatMs(ms) {
      if (ms >= timeUnits[7].factor) return (ms / timeUnits[7].factor).toFixed(ms >= 10 * timeUnits[7].factor ? 1 : 2) + ' yr';
      if (ms >= timeUnits[6].factor) return (ms / timeUnits[6].factor).toFixed(ms >= 10 * timeUnits[6].factor ? 1 : 2) + ' mo';
      if (ms >= timeUnits[4].factor) return (ms / timeUnits[4].factor).toFixed(ms >= 10 * timeUnits[4].factor ? 1 : 2) + ' d';
      if (ms >= timeUnits[3].factor) return (ms / timeUnits[3].factor).toFixed(ms >= 10 * timeUnits[3].factor ? 1 : 2) + ' h';
      if (ms >= timeUnits[2].factor) return (ms / timeUnits[2].factor).toFixed(ms >= 10 * timeUnits[2].factor ? 1 : 2) + ' min';
      if (ms >= timeUnits[1].factor) return (ms / timeUnits[1].factor).toFixed(ms >= 10 * timeUnits[1].factor ? 1 : 2) + ' s';
      if (ms >= 100) return ms.toFixed(0) + ' ms';
      if (ms >= 10) return ms.toFixed(1) + ' ms';
      return ms.toFixed(2) + ' ms';
    }

    function formatUnitValue(value, tickStep) {
      const digits = tickStep >= 10 ? 0 : tickStep >= 1 ? 1 : tickStep >= 0.1 ? 2 : 3;
      return value.toFixed(digits) + timeUnit.label;
    }

    function niceStep(rawValue) {
      if (rawValue <= 0) return 1;
      const magnitude = Math.pow(10, Math.floor(Math.log10(rawValue)));
      const scaled = rawValue / magnitude;
      let niceScaled = 1;
      if (scaled <= 1) niceScaled = 1;
      else if (scaled <= 2) niceScaled = 2;
      else if (scaled <= 5) niceScaled = 5;
      else niceScaled = 10;
      return niceScaled * magnitude;
    }

    function shortId(value, limit = 8) {
      if (value === null || value === undefined) return '';
      const text = String(value);
      return text.length <= limit ? text : text.slice(0, limit);
    }

    function visibleLanes() {
      return lanes
        .map((lane) => ({
          ...lane,
          visibleEvents: lane.events.filter((event) => activeStages.has(event.stage_id)),
        }))
        .filter((lane) => lane.visibleEvents.length > 0);
    }

    function ensureSelected(visibleLaneItems) {
      const visibleEvents = visibleLaneItems.flatMap((lane) => lane.visibleEvents);
      const visibleIds = new Set(visibleEvents.map((event) => event.event_id));
      if (!visibleEvents.length) {
        selectedEventId = null;
        return;
      }
      if (!selectedEventId || !visibleIds.has(selectedEventId)) {
        const preferred = visibleEvents.find((event) => event.event_style !== 'background');
        selectedEventId = (preferred || visibleEvents[0]).event_id;
      }
    }

    function renderCards() {
      const cardSpecs = [
        ['步骤窗口', formatMs(overview.window_ms || 0), (overview.lane_count || 0) + ' 条 lane'],
        ['轨迹数', String(overview.trajectory_count || 0), '最长 ' + formatMs(overview.longest_trajectory_ms || 0)],
        ['LLM 请求数', String(overview.llm_request_count || 0), (overview.prompt_tokens_total || 0).toLocaleString() + ' 个 prompt token'],
        ['工具调用数', String(overview.tool_call_count || 0), (overview.output_tokens_total || 0).toLocaleString() + ' 个输出 token'],
        ['进程数', String(overview.process_count || 0), (data.summary?.processes || []).join(', ') || 'n/a'],
        ['主导阶段', overview.top_stage_label || 'n/a', overview.top_stage_duration_ms ? formatMs(overview.top_stage_duration_ms) : ''],
      ];
      cards.innerHTML = '';
      for (const [label, value, meta] of cardSpecs) {
        const node = document.createElement('div');
        node.className = 'card';
        node.innerHTML =
          '<div class="card-label">' + label + '</div>' +
          '<div class="card-value">' + value + '</div>' +
          '<div class="card-meta">' + (meta || '&nbsp;') + '</div>';
        cards.appendChild(node);
      }
    }

    function renderLegend() {
      legend.innerHTML = '';
      for (const item of stageStats) {
        const label = document.createElement('label');
        label.className = 'legend-item';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = activeStages.has(item.stage_id);
        input.addEventListener('change', () => {
          if (input.checked) activeStages.add(item.stage_id);
          else activeStages.delete(item.stage_id);
          render();
        });
        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.background = stageColors[item.stage_id] || groupColors[item.stage_group] || '#475569';
        const text = document.createElement('span');
        text.textContent = item.stage_label + ' (' + item.span_count + ')';
        label.appendChild(input);
        label.appendChild(swatch);
        label.appendChild(text);
        legend.appendChild(label);
      }
    }

    function renderGlossary() {
      glossary.innerHTML = '';
      for (const item of stageStats) {
        const card = document.createElement('div');
        card.className = 'glossary-item';
        card.innerHTML =
          '<div class="glossary-head">' +
            '<div class="glossary-title">' + item.stage_label + '</div>' +
            '<div class="panel-note">' + formatMs(item.total_duration_ms) + '</div>' +
          '</div>' +
          '<div class="glossary-meta">' + item.stage_group + ' · ' + item.span_count + ' 个 span' + '</div>' +
          '<div class="glossary-meta">' + item.stage_description + '</div>' +
          '<div class="glossary-meta">原始 span: ' + item.raw_name_preview.join(', ') + '</div>';
        glossary.appendChild(card);
      }
    }

    function pickTrajectoryEvent(trajId) {
      const lane = lanes.find((item) => item.traj_id === trajId);
      if (!lane) return;
      const preferred = lane.events.find((event) => event.stage_id === 'trajectory_lifetime')
        || lane.events.find((event) => event.stage_id === 'llm_request')
        || lane.events[0];
      selectedEventId = preferred.event_id;
      render();
    }

    function renderTrajectoryCards() {
      trajectoryGrid.innerHTML = '';
      if (!trajectoryStats.length) {
        const empty = document.createElement('div');
        empty.className = 'panel-note';
        empty.textContent = '这个 step 没有可用的轨迹级 span。';
        trajectoryGrid.appendChild(empty);
        return;
      }
      for (const traj of trajectoryStats) {
        const card = document.createElement('div');
        card.className = 'trajectory-card';
        card.addEventListener('click', () => pickTrajectoryEvent(traj.traj_id));
        const topStages = (traj.top_stages || []).map((item) => item[0] + '×' + item[1]).join(' · ');
        card.innerHTML =
          '<div class="trajectory-title">' + (traj.mode === 'train' ? '训练' : traj.mode === 'val' ? '验证' : traj.mode) + ' · ' + traj.tag + ' · env ' + traj.env_id + '</div>' +
          '<div class="trajectory-meta">' +
            formatMs(traj.duration_ms) + ' · ' +
            traj.result + (traj.stop_reason ? ' / ' + traj.stop_reason : '') + ' · 轮次 ' + traj.turn_count +
            ' · 请求 ' + traj.request_count + ' · 工具 ' + traj.tool_call_count +
          '</div>' +
          '<div class="trajectory-meta">' +
            'prompt ' + traj.prompt_tokens_total.toLocaleString() + ' tok · 输出 ' + traj.output_tokens_total.toLocaleString() + ' tok' +
          '</div>' +
          '<div class="trajectory-meta">' + topStages + '</div>' +
          (traj.preview ? '<div class="trajectory-preview">' + traj.preview + '</div>' : '');
        trajectoryGrid.appendChild(card);
      }
    }

    function pickRequestEvent(request) {
      const lane = lanes.find((item) => item.traj_id === request.traj_id);
      if (!lane) return;
      const target = lane.events.find((event) => event.stage_id === 'llm_request' && event.sample_id === request.sample_id)
        || lane.events.find((event) => event.request_id && event.request_id === request.request_id);
      if (!target) return;
      selectedEventId = target.event_id;
      render();
    }

    function renderInferenceBreakdown() {
      inferenceCards.innerHTML = '';
      requestGrid.innerHTML = '';
      if (!(inferenceSummary.request_count > 0)) {
        const empty = document.createElement('div');
        empty.className = 'panel-note';
        empty.textContent = '这个 step 里没有可用的请求级推理 span。';
        requestGrid.appendChild(empty);
        return;
      }

      const cardsData = [
        ['Prefill 总耗时', formatMs(inferenceSummary.total_prefill_ms || 0), (inferenceSummary.prefill_share_pct || 0) + '% 的推理子阶段时间'],
        ['Decode 总耗时', formatMs(inferenceSummary.total_decode_ms || 0), (inferenceSummary.decode_share_pct || 0) + '% 的推理子阶段时间'],
        ['排队等待', formatMs(inferenceSummary.total_queue_ms || 0), (inferenceSummary.queue_share_pct || 0) + '% 的推理子阶段时间'],
        ['额外开销', formatMs(inferenceSummary.total_overhead_ms || 0), (inferenceSummary.overhead_share_pct || 0) + '% 的推理子阶段时间'],
        ['Prefill 效率', inferenceSummary.prefill_ms_per_1k_prompt_tokens !== null ? inferenceSummary.prefill_ms_per_1k_prompt_tokens + ' ms / 1k tok' : 'n/a', Math.round(inferenceSummary.avg_prompt_tokens || 0) + ' 个 prompt token / 请求'],
        ['Decode 效率', inferenceSummary.decode_ms_per_token !== null ? inferenceSummary.decode_ms_per_token + ' ms / tok' : 'n/a', Math.round(inferenceSummary.avg_output_tokens || 0) + ' 个输出 token / 请求'],
      ];
      for (const [label, value, meta] of cardsData) {
        const node = document.createElement('div');
        node.className = 'card';
        node.innerHTML =
          '<div class="card-label">' + label + '</div>' +
          '<div class="card-value">' + value + '</div>' +
          '<div class="card-meta">' + meta + '</div>';
        inferenceCards.appendChild(node);
      }

      for (const request of inferenceSummary.slow_requests || []) {
        const card = document.createElement('div');
        card.className = 'trajectory-card';
        card.addEventListener('click', () => pickRequestEvent(request));
        const turnLabel = request.env_step !== null && request.env_step !== undefined ? '第' + (Number(request.env_step) + 1) + '轮' : '轮次未知';
        const efficiency = [];
        if (request.prefill_ms_per_1k_prompt_tokens !== null) efficiency.push('prefill ' + request.prefill_ms_per_1k_prompt_tokens + ' ms/1k');
        if (request.decode_ms_per_token !== null) efficiency.push('decode ' + request.decode_ms_per_token + ' ms/tok');
        card.innerHTML =
          '<div class="trajectory-title">env ' + request.env_id + ' · ' + turnLabel + ' · req ' + shortId(request.request_id, 8) + '</div>' +
          '<div class="trajectory-meta">' +
            '总计 ' + formatMs(request.total_ms) + ' · 排队 ' + formatMs(request.queue_ms) +
            ' · prefill ' + formatMs(request.prefill_ms) +
            ' · decode ' + formatMs(request.decode_ms) +
            (request.overhead_ms ? ' · 开销 ' + formatMs(request.overhead_ms) : '') +
          '</div>' +
          '<div class="trajectory-meta">' +
            'prompt ' + request.prompt_tokens.toLocaleString() + ' tok · 输出 ' + request.output_tokens.toLocaleString() + ' tok' +
          '</div>' +
          '<div class="trajectory-meta">' + efficiency.join(' · ') + '</div>' +
          (request.preview ? '<div class="trajectory-preview">' + request.preview + '</div>' : '');
        requestGrid.appendChild(card);
      }
    }

    function renderNotes() {
      notes.innerHTML = '';
      const items = data.notes || [];
      if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'note';
        empty.textContent = '这个 step 没有额外的导出说明。';
        notes.appendChild(empty);
        return;
      }
      for (const item of items) {
        const node = document.createElement('div');
        node.className = 'note';
        node.textContent = item;
        notes.appendChild(node);
      }
    }

    function renderDetail() {
      if (!selectedEventId) {
        detailEmpty.hidden = false;
        detailContent.hidden = true;
        return;
      }
      const event = eventMap.get(selectedEventId);
      if (!event) {
        detailEmpty.hidden = false;
        detailContent.hidden = true;
        return;
      }
      detailEmpty.hidden = true;
      detailContent.hidden = false;
      detailTitle.textContent = event.title;
      const metaParts = [
        event.stage_label,
        event.process_label || 'process',
        '起点 ' + formatMs(event.start_offset_ms),
        '耗时 ' + formatMs(event.duration_ms),
      ];
      if (event.traj_id) metaParts.push('traj ' + shortId(event.traj_id, 14));
      if (event.request_id) metaParts.push('req ' + shortId(event.request_id, 8));
      detailSubtitle.textContent = metaParts.join(' · ');
      detailDescription.textContent = event.stage_description;

      detailBadges.innerHTML = '';
      const badges = [
        ['分组', event.stage_group],
        ['原始名', event.raw_name],
        ['样式', event.event_style],
      ];
      if (event.env_step !== null && event.env_step !== undefined) badges.push(['轮次', String(Number(event.env_step) + 1)]);
      if (event.attrs?.prompt_tokens) badges.push(['Prompt', String(event.attrs.prompt_tokens) + ' tok']);
      if (event.attrs?.output_tokens) badges.push(['输出', String(event.attrs.output_tokens) + ' tok']);
      if (event.request_prompt_tokens && !event.attrs?.prompt_tokens) badges.push(['Prompt', String(event.request_prompt_tokens) + ' tok']);
      if (event.request_output_tokens && !event.attrs?.output_tokens) badges.push(['输出', String(event.request_output_tokens) + ' tok']);
      if (event.performance_label && event.performance_value !== null && event.performance_value !== undefined) {
        badges.push([event.performance_label, String(event.performance_value) + ' ' + event.performance_unit]);
      }
      if (event.attrs?.mode) badges.push(['模式', String(event.attrs.mode)]);
      if (event.attrs?.result) badges.push(['结果', String(event.attrs.result)]);
      if (event.attrs?.stop_reason) badges.push(['结束原因', String(event.attrs.stop_reason)]);
      for (const [label, value] of badges) {
        const chip = document.createElement('div');
        chip.className = 'badge';
        chip.textContent = label + ': ' + value;
        detailBadges.appendChild(chip);
      }

      if (event.preview) {
        detailPreview.hidden = false;
        detailPreview.textContent = event.preview;
      } else {
        detailPreview.hidden = true;
        detailPreview.textContent = '';
      }

      detailAttrs.innerHTML = '';
      const attrs = event.attrs || {};
      const preferredKeys = ['message_count', 'message_roles', 'last_message_role', 'last_message_preview', 'response_kind', 'response_preview', 'request_id', 'prompt_tokens', 'output_tokens', 'finish_reasons', 'env_step', 'mode', 'tag', 'result', 'stop_reason', 'source'];
      const keys = [...preferredKeys.filter((key) => key in attrs), ...Object.keys(attrs).sort().filter((key) => !preferredKeys.includes(key))];
      for (const key of keys) {
        const value = attrs[key];
        if (value === null || value === undefined || value === '') continue;
        const item = document.createElement('div');
        item.className = 'detail-attr';
        const rendered = typeof value === 'object' ? JSON.stringify(value) : String(value);
        item.innerHTML = '<div class="detail-key">' + key + '</div><div>' + rendered + '</div>';
        detailAttrs.appendChild(item);
      }
      detailJson.textContent = JSON.stringify(event, null, 2);
    }

    function clampTooltip(x, y) {
      const padding = 14;
      const rect = tooltip.getBoundingClientRect();
      tooltip.style.left = Math.max(padding, Math.min(x, window.innerWidth - rect.width - padding)) + 'px';
      tooltip.style.top = Math.max(padding, Math.min(y, window.innerHeight - rect.height - padding)) + 'px';
    }

    function showTooltip(evt, event) {
      const lines = [
        event.title,
        event.stage_label + ' · ' + event.stage_group,
        '起点=' + formatMs(event.start_offset_ms) + ' · 耗时=' + formatMs(event.duration_ms),
      ];
      if (event.traj_id) lines.push('traj=' + shortId(event.traj_id, 18));
      if (event.request_id) lines.push('req=' + shortId(event.request_id, 10));
      if (event.request_prompt_tokens) lines.push('prompt=' + event.request_prompt_tokens + ' tok');
      if (event.request_output_tokens) lines.push('output=' + event.request_output_tokens + ' tok');
      if (event.performance_label && event.performance_value !== null && event.performance_value !== undefined) {
        lines.push(event.performance_label + '=' + event.performance_value + ' ' + event.performance_unit);
      }
      if (event.preview) lines.push(event.preview);
      tooltip.textContent = lines.join('\\n');
      tooltip.style.opacity = '1';
      clampTooltip(evt.clientX + 12, evt.clientY + 12);
    }

    function hideTooltip() {
      tooltip.style.opacity = '0';
    }

    function renderTimeline(visibleLaneItems) {
      const pxPerUnit = Number(zoom.value);
      zoomLabel.firstChild.textContent = '缩放（' + timeUnit.label + '）';

      const leftWidth = 330;
      const topPadding = 56;
      const laneHeight = 42;
      const laneGap = 10;
      const barHeight = 18;
      const chartWidth = leftWidth + Math.max(900, Math.ceil(windowUnits * pxPerUnit) + 120);
      const chartHeight = topPadding + Math.max(1, visibleLaneItems.length) * (laneHeight + laneGap) + 34;

      chart.setAttribute('width', chartWidth);
      chart.setAttribute('height', chartHeight);
      chart.innerHTML = '';

      const tickStep = niceStep(90 / pxPerUnit);
      for (let tick = 0, value = 0; value <= windowUnits + tickStep * 0.5; tick += 1, value += tickStep) {
        const x = leftWidth + value * pxPerUnit;
        const grid = createSvg('line');
        grid.setAttribute('x1', x);
        grid.setAttribute('x2', x);
        grid.setAttribute('y1', topPadding - 22);
        grid.setAttribute('y2', chartHeight);
        grid.setAttribute('stroke', tick % 5 === 0 ? '#c9b99f' : '#e7ded1');
        grid.setAttribute('stroke-width', tick % 5 === 0 ? '1.1' : '0.8');
        chart.appendChild(grid);

        const label = createSvg('text');
        label.setAttribute('x', x + 3);
        label.setAttribute('y', 26);
        label.setAttribute('class', 'axis-label');
        label.textContent = formatUnitValue(value, tickStep);
        chart.appendChild(label);
      }

      const divider = createSvg('line');
      divider.setAttribute('x1', leftWidth);
      divider.setAttribute('x2', leftWidth);
      divider.setAttribute('y1', topPadding - 20);
      divider.setAttribute('y2', chartHeight);
      divider.setAttribute('stroke', '#d1c7b7');
      divider.setAttribute('stroke-width', '1');
      chart.appendChild(divider);

      if (!visibleLaneItems.length) {
        const text = createSvg('text');
        text.setAttribute('x', leftWidth + 20);
        text.setAttribute('y', topPadding + 18);
        text.setAttribute('class', 'lane-meta');
        text.textContent = '当前没有可见事件，请在图例里重新勾选阶段。';
        chart.appendChild(text);
        return;
      }

      for (let index = 0; index < visibleLaneItems.length; index += 1) {
        const lane = visibleLaneItems[index];
        const y = topPadding + index * (laneHeight + laneGap);

        const bg = createSvg('rect');
        bg.setAttribute('x', 0);
        bg.setAttribute('y', y - 8);
        bg.setAttribute('width', chartWidth);
        bg.setAttribute('height', laneHeight + laneGap);
        bg.setAttribute('fill', index % 2 === 0 ? '#fffdf8' : '#fbf7ef');
        chart.appendChild(bg);

        const title = createSvg('text');
        title.setAttribute('x', 14);
        title.setAttribute('y', y + 6);
        title.setAttribute('class', 'lane-title');
        title.textContent = lane.lane_label;
        chart.appendChild(title);

        const meta = createSvg('text');
        meta.setAttribute('x', 14);
        meta.setAttribute('y', y + 21);
        meta.setAttribute('class', 'lane-meta');
        meta.textContent = lane.lane_meta;
        chart.appendChild(meta);

        for (const event of lane.visibleEvents) {
          const x = leftWidth + (event.start_offset_ms / timeUnit.factor) * pxPerUnit;
          const width = Math.max(1.8, (event.duration_ms / timeUnit.factor) * pxPerUnit);
          const rect = createSvg('rect');
          rect.setAttribute('x', x);
          rect.setAttribute('y', y + 7);
          rect.setAttribute('width', width);
          rect.setAttribute('height', barHeight);
          rect.setAttribute('rx', '9');
          rect.setAttribute('fill', resolveColor(event));
          rect.setAttribute('opacity', event.event_style === 'background' ? '0.22' : event.event_style === 'muted' ? '0.5' : '0.92');
          if (event.event_id === selectedEventId) {
            rect.setAttribute('stroke', '#3f1d12');
            rect.setAttribute('stroke-width', '1.4');
          }
          rect.addEventListener('mousemove', (evt) => showTooltip(evt, event));
          rect.addEventListener('mouseleave', hideTooltip);
          rect.addEventListener('click', () => {
            selectedEventId = event.event_id;
            render();
          });
          chart.appendChild(rect);

          if (width >= 66 && event.event_style === 'standard') {
            const label = createSvg('text');
            label.setAttribute('x', x + 8);
            label.setAttribute('y', y + 20);
            label.setAttribute('class', 'event-label');
            label.textContent = event.stage_label;
            chart.appendChild(label);
          }
        }
      }
    }

    function render() {
      const visibleLaneItems = visibleLanes();
      ensureSelected(visibleLaneItems);
      subtitle.textContent =
        'step ' + data.step +
        ' · span 数 ' + (data.summary?.total_spans || 0) +
        ' · 可见 lane ' + visibleLaneItems.length + '/' + (lanes.length || 0) +
        ' · 时间窗口 ' + formatMs(windowMs) +
        ((data.summary?.dropped_spans || 0) > 0 ? ' · 过滤异常 span ' + data.summary.dropped_spans + ' 个' : '');
      renderTimeline(visibleLaneItems);
      renderDetail();
    }

    const defaultZoom = Math.max(0.4, Math.min(72, 2400 / windowUnits));
    zoom.value = defaultZoom.toFixed(1);
    zoom.addEventListener('input', render);
    window.addEventListener('resize', hideTooltip);
    window.addEventListener('scroll', hideTooltip);

    renderCards();
    renderLegend();
    renderGlossary();
    renderInferenceBreakdown();
    renderTrajectoryCards();
    renderNotes();
    render();
  </script>
</body>
</html>
"""
    return template.replace("__STEP__", str(bundle["step"])).replace("__DATA_JSON__", data_json)


def export_trace_step(trace_dir: str, step: int, max_exported_spans: int = 20000) -> str:
    trace_path = Path(trace_dir)
    spans = _load_step_spans(trace_dir, step)
    if len(spans) > max_exported_spans:
        logger.warning(
            "Skipping trace HTML export for step %s because span count %s exceeds limit %s.",
            step,
            len(spans),
            max_exported_spans,
        )
        return ""

    bundle = _build_step_bundle(spans, step)
    timeline_dir = trace_path / "timeline" / "steps"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    json_path = timeline_dir / f"step_{step:06d}.json"
    html_path = timeline_dir / f"step_{step:06d}.html"
    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_step_html(bundle), encoding="utf-8")
    _export_index(trace_dir)
    return str(html_path)


def _format_duration_ms(duration_ms: float) -> str:
    units = (
        ("yr", 365.25 * 24 * 60 * 60 * 1000),
        ("mo", 30.4375 * 24 * 60 * 60 * 1000),
        ("d", 24 * 60 * 60 * 1000),
        ("h", 60 * 60 * 1000),
        ("min", 60 * 1000),
        ("s", 1000),
    )
    for label, factor in units:
        if duration_ms >= factor:
            precision = 1 if duration_ms >= factor * 10 else 2
            return f"{duration_ms / factor:.{precision}f}{label}"
    if duration_ms >= 100:
        return f"{duration_ms:.0f}ms"
    if duration_ms >= 10:
        return f"{duration_ms:.1f}ms"
    return f"{duration_ms:.2f}ms"


def _export_index(trace_dir: str) -> None:
    trace_path = Path(trace_dir)
    timeline_dir = trace_path / "timeline" / "steps"
    step_items = []
    for json_path in sorted(timeline_dir.glob("step_*.json")):
        bundle = json.loads(json_path.read_text(encoding="utf-8"))
        overview = bundle.get("overview", {})
        step_items.append(
            {
                "step": bundle["step"],
                "window_ms": bundle["summary"]["window_ms"],
                "total_spans": bundle["summary"]["total_spans"],
                "window_label": _format_duration_ms(bundle["summary"]["window_ms"]),
                "html_path": f"steps/step_{bundle['step']:06d}.html",
                "trajectory_count": overview.get("trajectory_count", 0),
                "llm_request_count": overview.get("llm_request_count", 0),
                "tool_call_count": overview.get("tool_call_count", 0),
                "output_tokens_total": overview.get("output_tokens_total", 0),
                "top_stage_label": overview.get("top_stage_label", ""),
            }
        )
    index_path = trace_path / "timeline" / "index.html"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"<tr>"
        f"<td>{item['step']}</td>"
        f"<td>{item['total_spans']}</td>"
        f"<td title=\"{item['window_ms']} ms\">{item['window_label']}</td>"
        f"<td>{item['trajectory_count']}</td>"
        f"<td>{item['llm_request_count']}</td>"
        f"<td>{item['tool_call_count']}</td>"
        f"<td>{item['output_tokens_total']}</td>"
        f"<td>{item['top_stage_label'] or 'n/a'}</td>"
        f"<td><a href=\"{item['html_path']}\">open</a></td>"
        f"</tr>"
        for item in reversed(step_items)
    )
    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ROLL Trace 索引</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --border: #d1c7b7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(194, 65, 12, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f2e9 0%, #efe7da 100%);
    }}
    .card {{
      max-width: 1120px;
      margin: 0 auto;
      border: 1px solid var(--border);
      border-radius: 20px;
      background: rgba(255, 253, 248, 0.94);
      box-shadow: 0 20px 60px rgba(36, 27, 18, 0.10);
      overflow: hidden;
    }}
    .header {{
      padding: 18px 20px 12px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 253, 248, 0.94);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: -0.02em;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 13px 16px;
      border-bottom: 1px solid #e4dcc8;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #f1ecdf;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    tr:nth-child(even) td {{
      background: rgba(251, 247, 239, 0.7);
    }}
    a {{
      color: #9a3412;
      text-decoration: none;
      font-weight: 600;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .note {{
      padding: 0 20px 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1>ROLL Trace 索引</h1>
      <p>生成目录：<code>{trace_dir}</code>。每一行对应一个已导出的 step，展示规范化阶段名和轨迹级统计。</p>
    </div>
    <table>
      <thead><tr><th>Step</th><th>Span 数</th><th>时间窗口</th><th>轨迹数</th><th>请求数</th><th>工具数</th><th>输出 Token</th><th>主导阶段</th><th>时间线</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="note">单个 step 内的阶段耗时可能重叠，因为 trace 保留的是 inclusive span。进入具体 step 页面可以查看中文阶段说明、轨迹摘要以及事件级元数据。</div>
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def export_trace_directory(trace_dir: str, max_exported_spans: int = 20000) -> list[str]:
    raw_steps_dir = Path(trace_dir) / "raw" / "steps"
    outputs = []
    if not raw_steps_dir.exists():
        return outputs
    for step_dir in sorted(raw_steps_dir.iterdir()):
        if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
            continue
        step = int(step_dir.name.split("_")[-1])
        html_path = export_trace_step(trace_dir, step, max_exported_spans=max_exported_spans)
        if html_path:
            outputs.append(html_path)
    return outputs
