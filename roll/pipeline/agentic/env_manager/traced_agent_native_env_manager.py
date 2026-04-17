import copy
import json
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import ray
import torch
from codetiming import Timer
from tensordict import TensorDict

from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.token_mask_utils import convert_list_content_str
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.utils.constants import GenerateStopReason, EpisodeStopReason
from roll.utils.functionals import pad_to_length, aggregate_metrics
from roll.utils.hash_utils import compute_object_hash
from roll.utils.tracing import get_trace_manager


class TracedAgentNativeStepEnvManager(TrajEnvManager):
    """
    Traced variant of AgentNativeStepEnvManager with structured tracing spans.
    Used for swe/tb native env.
    """
    log_stats: Dict
    failure_mode: str
    env_reset_failed: bool
    stop_reason: EpisodeStopReason
    tools: List[Dict]
    traj_start_time: float

    @staticmethod
    def _normalize_preview(value: Any, limit: int = 240) -> str:
        if value is None:
            return ""
        normalized = " ".join(str(value).split())
        return normalized[:limit]

    @staticmethod
    def _make_sample_id(traj_id: Optional[str], env_step: int) -> Optional[str]:
        if traj_id is None:
            return None
        return f"{traj_id}:step:{env_step}"

    def _current_traj_step(self, rollout_cache: Optional[RolloutCache] = None) -> int:
        cache = rollout_cache if rollout_cache is not None else getattr(self, "rollout_cache", None)
        if cache is None:
            return 0
        return int(getattr(cache, "step", 0))

    def _build_step_trace_attrs(self, *, episode_id: Optional[int], traj_id: Optional[str], env_step: int) -> dict[str, Any]:
        return {
            "env_id": self.env_config["env_id"],
            "tag": self.env_config["tag"],
            "group_id": self.env_config["group_id"],
            "mode": getattr(self, "mode", "train"),
            "env_step": int(env_step),
            "train_step": int(self.current_step),
            "episode_id": episode_id,
            "traj_id": traj_id,
            "sample_id": self._make_sample_id(traj_id, env_step),
        }

    @staticmethod
    def _classify_step_kind(response: Any, tool_names: list[str], terminated: bool) -> str:
        if tool_names:
            return "tool_call"
        if isinstance(response, EpisodeStopReason):
            return "control"
        if terminated:
            return "terminal_response"
        return "assistant_response"

    @staticmethod
    def _get_completed_step_record(rollout_cache: RolloutCache) -> dict[str, Any]:
        if len(rollout_cache.history) >= 2:
            return rollout_cache.history[-2]
        if rollout_cache.history:
            return rollout_cache.history[-1]
        return {}

    @staticmethod
    def _apply_step_outcome_attrs(span: Any, step_record: dict[str, Any]) -> None:
        if span is None:
            return
        for key in (
            "reward",
            "failure_mode",
            "stop_reason",
            "action_is_valid",
            "step_kind",
            "request_id",
        ):
            if key in step_record:
                span.set_attribute(key, step_record.get(key))
        if "metrics" in step_record and isinstance(step_record["metrics"], dict):
            raw_reward = step_record["metrics"].get("raw_reward")
            success = step_record["metrics"].get("success")
            if raw_reward is not None:
                span.set_attribute("raw_reward", raw_reward)
            if success is not None:
                span.set_attribute("success", success)
        if "llm_response" in step_record:
            span.set_attribute("response_preview", TracedAgentNativeStepEnvManager._normalize_preview(step_record["llm_response"]))
        span.set_attribute("terminated", step_record.get("terminated", False))
        span.set_attribute("truncated", step_record.get("truncated", False))

    @staticmethod
    def _extract_tool_call_names(response: Any) -> list[str]:
        if not isinstance(response, str):
            return []

        tool_names: list[str] = []

        def _append_xml_function_names(text: str) -> None:
            for name in re.findall(r"<function\s*=\s*([^>]+)>", text, flags=re.DOTALL):
                normalized = name.strip()
                if normalized:
                    tool_names.append(normalized)

        if "<tool_call>" in response:
            tool_call_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL)
            for block in tool_call_blocks:
                if "<function" in block:
                    _append_xml_function_names(block)
                    continue
                try:
                    tool_call_payload = json.loads(block.strip())
                except json.JSONDecodeError:
                    continue
                tool_name = str(tool_call_payload.get("name", "")).strip()
                if tool_name:
                    tool_names.append(tool_name)
        elif "<function" in response:
            _append_xml_function_names(response)

        return tool_names

    @staticmethod
    def _estimate_inference_breakdown(
        total_duration_ms: float,
        prompt_tokens: int,
        output_tokens: int,
    ) -> tuple[float, float, float]:
        total_duration_ms = max(float(total_duration_ms), 0.0)
        if total_duration_ms <= 0:
            return 0.0, 0.0, 0.0

        if prompt_tokens <= 0 and output_tokens <= 0:
            return 0.0, 0.0, total_duration_ms

        if prompt_tokens > 0 and output_tokens <= 0:
            return total_duration_ms, 0.0, 0.0

        if prompt_tokens <= 0 and output_tokens > 0:
            decode_ms = total_duration_ms * 0.95
            return 0.0, decode_ms, max(total_duration_ms - decode_ms, 0.0)

        overhead_share = 0.05
        available_ms = total_duration_ms * (1.0 - overhead_share)
        prompt_weight = float(prompt_tokens) * 1.15
        decode_weight = float(output_tokens)
        total_weight = prompt_weight + decode_weight
        if total_weight <= 0:
            return 0.0, 0.0, total_duration_ms

        prefill_ms = available_ms * (prompt_weight / total_weight)
        decode_ms = available_ms - prefill_ms
        overhead_ms = max(total_duration_ms - prefill_ms - decode_ms, 0.0)
        return prefill_ms, decode_ms, overhead_ms

    @staticmethod
    def _coerce_phase_duration_ms(value: Any, unit: str) -> Optional[float]:
        if value is None:
            return None
        duration = float(value)
        if unit == "seconds":
            duration *= 1000.0
        return max(duration, 0.0)

    def _extract_vllm_phase_breakdown_ms(self, phase_timing: Any) -> Optional[dict[str, Any]]:
        if not isinstance(phase_timing, dict):
            return None

        unit = str(phase_timing.get("unit", "seconds"))
        return {
            "source": str(phase_timing.get("source", "vllm_phase_timing")),
            "queue_time_ms": self._coerce_phase_duration_ms(
                phase_timing.get("queue_time", phase_timing.get("queued_time")),
                unit,
            ),
            "prefill_time_ms": self._coerce_phase_duration_ms(phase_timing.get("prefill_time"), unit),
            "decode_time_ms": self._coerce_phase_duration_ms(phase_timing.get("decode_time"), unit),
            "inference_time_ms": self._coerce_phase_duration_ms(phase_timing.get("inference_time"), unit),
            "e2e_time_ms": self._coerce_phase_duration_ms(phase_timing.get("e2e_time"), unit),
            "ttft_ms": self._coerce_phase_duration_ms(phase_timing.get("time_to_first_token"), unit),
        }

    def _record_inference_phase_spans(
        self,
        tracer,
        request_span,
        *,
        trace_attrs: Dict[str, Any],
        sample_id: Optional[str],
        traj_id: Optional[str],
        request_id: str,
        prompt_tokens: int,
        output_tokens: int,
        response_preview: str,
        phase_timing: Any,
    ) -> None:
        if request_span.start_wall_ns is None or request_span.end_wall_ns is None or request_span.duration_ms is None:
            return

        phase_breakdown = self._extract_vllm_phase_breakdown_ms(phase_timing)
        if phase_breakdown is None:
            prefill_ms, decode_ms, overhead_ms = self._estimate_inference_breakdown(
                total_duration_ms=request_span.duration_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            queue_ms = 0.0
            inference_ms = prefill_ms + decode_ms
            e2e_ms = request_span.duration_ms
            ttft_ms = None
            timing_source = "fallback_local_timing"
        else:
            queue_ms = phase_breakdown.get("queue_time_ms") or 0.0
            prefill_ms = phase_breakdown.get("prefill_time_ms") or 0.0
            decode_ms = phase_breakdown.get("decode_time_ms") or 0.0
            inference_ms = phase_breakdown.get("inference_time_ms")
            if inference_ms is None:
                inference_ms = prefill_ms + decode_ms
            e2e_ms = phase_breakdown.get("e2e_time_ms")
            if e2e_ms is None:
                e2e_ms = request_span.duration_ms
            ttft_ms = phase_breakdown.get("ttft_ms")
            timing_source = phase_breakdown["source"]

            consumed_ms = queue_ms + prefill_ms + decode_ms
            overhead_ms = max(request_span.duration_ms - consumed_ms, 0.0)

        segments: list[tuple[str, str, str, float, str]] = []
        if queue_ms > 0:
            segments.append(("rollout.wait_worker", "rollout", "rollout", queue_ms, timing_source))
        if prefill_ms > 0:
            segments.append(("inference.prefill", "inference", "inference", prefill_ms, timing_source))
        if decode_ms > 0:
            segments.append(("inference.decode", "inference", "inference", decode_ms, timing_source))
        if overhead_ms > 0 or not segments:
            overhead_source = timing_source if phase_breakdown is None else "request_residual_local"
            segments.append(("inference.overhead", "inference", "inference", overhead_ms, overhead_source))

        shared_attrs = {
            **trace_attrs,
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "response_preview": response_preview,
            "timing_source": timing_source,
        }
        trace_context = request_span.child_context(sample_id=sample_id, traj_id=traj_id)
        cursor_ns = int(request_span.start_wall_ns)
        request_end_ns = int(request_span.end_wall_ns)
        for index, (span_name, phase, category, duration_ms, source) in enumerate(segments):
            if index == len(segments) - 1:
                segment_end_ns = request_end_ns
            else:
                segment_end_ns = min(request_end_ns, cursor_ns + int(round(duration_ms * 1_000_000)))
            if segment_end_ns <= cursor_ns:
                continue
            tracer.record_completed_span(
                span_name,
                phase=phase,
                category=category,
                trace_context=trace_context,
                sample_id=sample_id,
                traj_id=traj_id,
                attrs={**shared_attrs, "source": source},
                start_time_ns=cursor_ns,
                end_time_ns=segment_end_ns,
            )
            cursor_ns = segment_end_ns

        metrics_end_ns = request_end_ns
        metrics_start_ns = max(int(request_span.start_wall_ns), metrics_end_ns - 1_000)
        if metrics_end_ns > metrics_start_ns:
            tracer.record_completed_span(
                "inference.metrics",
                phase="inference",
                category="inference",
                trace_context=trace_context,
                sample_id=sample_id,
                traj_id=traj_id,
                attrs={
                    **shared_attrs,
                    "source": timing_source,
                    "queue_time_ms": round(queue_ms, 6),
                    "prefill_time_ms": round(prefill_ms, 6),
                    "decode_time_ms": round(decode_ms, 6),
                    "inference_time_ms": round(float(inference_ms or 0.0), 6),
                    "e2e_time_ms": round(float(e2e_ms or request_span.duration_ms), 6),
                    "overhead_time_ms": round(overhead_ms, 6),
                    "ttft_ms": round(float(ttft_ms), 6) if ttft_ms is not None else None,
                },
                start_time_ns=metrics_start_ns,
                end_time_ns=metrics_end_ns,
            )

    def run_rollout_loop(self, data: DataProto):
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        env_id = self.env_config['env_id']
        tag = self.env_config['tag']
        group_id = self.env_config['group_id']
        mode = getattr(self, 'mode', 'train')

        tracer = get_trace_manager(component=f"env_{env_id}")
        _base_attrs = {"env_id": env_id, "tag": tag, "group_id": group_id, "mode": mode}

        def _make_traj_id(episode_id):
            return f"{tag}_{group_id}_{episode_id}_{env_id}"

        with Timer(name="reset", logger=None) as reset_timer:
            _reset_span = tracer.span("env.reset", phase="env", attrs={**_base_attrs})
            _reset_span.__enter__()
            rollout_cache: RolloutCache = self.reset()
            # Add episode_id to reset span after reset() completes
            if hasattr(self, 'episode_id') and self.episode_id is not None:
                _reset_span.set_attribute("episode_id", self.episode_id)
                _reset_span.set_attribute("traj_id", _make_traj_id(self.episode_id))
            _reset_span.__exit__(None, None, None)
        self.log_stats["reset_time"] = round(reset_timer.last, 4)
        start_step = self.current_step
        max_reset_retries = 0
        trajectory_span = None
        try:
            while self.running and rollout_cache is not None:
                if self.env_reset_failed:
                    max_reset_retries += 1
                    self.logger.error(f"[ROLLOUT_LOOP] Failed! - due to sandbox initialization failure...")
                    if trajectory_span is not None:
                        trajectory_span.set_attribute("result", "env_reset_failed")
                        trajectory_span.set_attribute("stop_reason", self.stop_reason.name)
                        trajectory_span.__exit__(None, None, None)
                        trajectory_span = None

                    rollout: DataProto = self.create_placeholder_rollout(self.episode_id)
                    rollout.meta_info["drop_flag"] = True
                    failed_traj_id = _make_traj_id(self.episode_id) if getattr(self, 'episode_id', None) is not None else None
                    _put_span = tracer.span(
                        "rollout.put_batch",
                        phase="rollout",
                        attrs={**_base_attrs, "episode_id": self.episode_id, "traj_id": failed_traj_id, "drop_flag": True},
                        traj_id=failed_traj_id,
                    )
                    _put_span.__enter__()
                    ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout, self.env_config['env_id']))
                    _put_span.__exit__(None, None, None)

                    _close_span = tracer.span(
                        "env.close",
                        phase="env",
                        attrs={**_base_attrs, "episode_id": self.episode_id, "traj_id": failed_traj_id},
                        traj_id=failed_traj_id,
                    )
                    _close_span.__enter__()
                    self.env.close()
                    _close_span.__exit__(None, None, None)
                    if max_reset_retries > 3:
                        backoff_time = min(3600, 10 * max_reset_retries)
                        self.logger.warning(f"[ROLLOUT_LOOP] Avoidance mode - Backing off for {backoff_time}s (retry #{max_reset_retries})")
                        time.sleep(backoff_time)
                    else:
                        time.sleep(10)
                    with Timer(name="reset", logger=None) as reset_timer:
                        _reset_span = tracer.span("env.reset", phase="env", attrs={**_base_attrs})
                        _reset_span.__enter__()
                        rollout_cache = self.reset()
                        if hasattr(self, 'episode_id') and self.episode_id is not None:
                            _reset_span.set_attribute("episode_id", self.episode_id)
                            _reset_span.set_attribute("traj_id", _make_traj_id(self.episode_id))
                        _reset_span.__exit__(None, None, None)
                    self.log_stats["reset_time"] = round(reset_timer.last, 4)
                    start_step = self.current_step
                    continue

                max_reset_retries = 0
                episode_id = getattr(self, 'episode_id', None)
                traj_id = _make_traj_id(episode_id) if episode_id is not None else None
                if trajectory_span is None:
                    trajectory_span = tracer.span(
                        "trajectory.lifetime",
                        phase="trajectory",
                        attrs={**_base_attrs, "episode_id": episode_id, "traj_id": traj_id},
                        traj_id=traj_id,
                    )
                    trajectory_span.__enter__()

                traj_env_step = self._current_traj_step(rollout_cache)
                step_attrs = self._build_step_trace_attrs(
                    episode_id=episode_id,
                    traj_id=traj_id,
                    env_step=traj_env_step,
                )
                sample_id = step_attrs["sample_id"]
                _trajectory_step_span = tracer.span(
                    "trajectory.step",
                    phase="trajectory",
                    attrs=step_attrs,
                    traj_id=traj_id,
                    sample_id=sample_id,
                )
                _trajectory_step_span.__enter__()
                try:
                    with Timer(name="generate", logger=None) as generate_timer:
                        _gen_span = tracer.span(
                            "inference.generate",
                            phase="inference",
                            attrs=step_attrs,
                            traj_id=traj_id,
                            sample_id=sample_id,
                        )
                        _gen_span.__enter__()
                        lm_output: DataProto = self.make_decision(
                            rollout_cache,
                            tracer=tracer,
                            trace_context=_gen_span.child_context(sample_id=sample_id, traj_id=traj_id),
                            trace_attrs=step_attrs,
                            sample_id=sample_id,
                            traj_id=traj_id,
                        )
                        stop_reason = lm_output.meta_info.pop("stop_reason")
                        if stop_reason == GenerateStopReason.MAX_LENGTH:
                            self.stop_reason = EpisodeStopReason.MAX_LENGTH
                        elif stop_reason == GenerateStopReason.ABORT:
                            self.stop_reason = EpisodeStopReason.ABORT
                        request_id = lm_output.meta_info.get("request_id") or lm_output.meta_info.get("trace_request_id")
                        if request_id:
                            _trajectory_step_span.set_attribute("request_id", request_id)
                        _gen_span.__exit__(None, None, None)
                    self.log_stats["current_step"].append(self.current_step)
                    self.log_stats["generate_time"].append(round(generate_timer.last))

                    with Timer(name="step", logger=None) as step_timer:
                        if stop_reason in [GenerateStopReason.FINISH, GenerateStopReason.MAX_LENGTH]:
                            _env_step_span = tracer.span(
                                "env.step",
                                phase="env",
                                attrs={**step_attrs, **({"request_id": request_id} if request_id else {})},
                                traj_id=traj_id,
                                sample_id=sample_id,
                            )
                            _env_step_span.__enter__()
                            rollout_cache = self.step(lm_output)
                            completed_step = self._get_completed_step_record(rollout_cache)
                            self._apply_step_outcome_attrs(_env_step_span, completed_step)
                            self._apply_step_outcome_attrs(_trajectory_step_span, completed_step)
                            _env_step_span.__exit__(None, None, None)
                    self.log_stats["step_time"].append(round(step_timer.last, 4))
                finally:
                    _trajectory_step_span.__exit__(None, None, None)

                if self.running and rollout_cache.terminated:
                    rollout = self.formulate_rollouts(rollout_cache)
                    traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                    traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                    rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                    rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                    _put_span = tracer.span(
                        "rollout.put_batch",
                        phase="rollout",
                        attrs={
                            **_base_attrs,
                            "episode_id": self.episode_id,
                            "traj_id": traj_id,
                            "start_step": start_step,
                            "sample_count": rollout.batch.batch_size[0],
                        },
                        traj_id=traj_id,
                    )
                    _put_span.__enter__()
                    ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout, self.env_config['env_id']))
                    _put_span.__exit__(None, None, None)

                    if trajectory_span is not None:
                        trajectory_span.set_attribute("result", "terminated" if rollout_cache.terminated else "truncated")
                        trajectory_span.set_attribute("stop_reason", self.stop_reason.name)
                        trajectory_span.__exit__(None, None, None)
                        trajectory_span = None

                    rollout_cache = self.reset()
                    start_step = self.current_step
        finally:
            if trajectory_span is not None:
                trajectory_span.set_attribute("result", "stopped")
                trajectory_span.set_attribute("stop_reason", self.stop_reason.name)
                trajectory_span.__exit__(None, None, None)

        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, None, self.env_config['env_id']))

    def reset(self) -> Optional[RolloutCache]:
        self.log_stats = {"generate_time": [], "step_time": [], "current_step": [], "reset_time": 0.0, "response_length": [], "tokens_per_second": []}
        self.stop_reason = EpisodeStopReason.FINISH
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(
            self.env_config['group_id'],
            self.env_config['env_id']
        ))
        if self.episode_id is None:
            assert not self.running
            return None

        seed = self.group_seed + self.episode_id
        self.traj_start_time = time.time()
        observation, info = self.env.reset(seed=seed)
        if observation is None:
            return None

        if self.env.env_reset_failed:
            self.env_reset_failed = True
            self.logger.error(f"[ENV_RESET] Failed! - Environment reset failed, observation: {json.dumps(observation, ensure_ascii=False)}, env_reset_failed: {self.env.env_reset_failed}")
            self.failure_mode = info.get("failure_mode", "Sandbox Initialization Failed")
            self.stop_reason = EpisodeStopReason.ENV_RESET_FAILED
        else:
            self.env_reset_failed = False

        self.tools = info.get("tools", [])
        self.rollout_cache.history.append({
            "observation": copy.deepcopy(observation),
            "messages": None,     # agent input messages
            **info,
        })
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        if llm_output.batch is not None:
            response = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=False)[0]
        else:
            response = self.stop_reason
        tracer = get_trace_manager(component=f"env_{self.env_config['env_id']}")
        sample_id = llm_output.meta_info.get("trace_sample_id")
        traj_id = llm_output.meta_info.get("trace_traj_id")
        request_id = llm_output.meta_info.get("request_id") or llm_output.meta_info.get("trace_request_id")
        traj_env_step = llm_output.meta_info.get("trace_env_step", self._current_traj_step())
        tool_names = self._extract_tool_call_names(response)
        if tool_names:
            tool_attrs = {
                "env_id": self.env_config['env_id'],
                "tag": self.env_config['tag'],
                "group_id": self.env_config['group_id'],
                "mode": getattr(self, 'mode', 'train'),
                "env_step": traj_env_step,
                "train_step": self.current_step,
                "episode_id": getattr(self, 'episode_id', None),
                "traj_id": traj_id,
                "sample_id": sample_id,
                "request_id": request_id,
                "tool_name": tool_names[0],
                "tool_names": tool_names,
                "tool_call_count": len(tool_names),
                "response_preview": self._normalize_preview(response),
                "source": "env_tool_runtime",
            }
            _tool_span = tracer.span(
                "trajectory.tool_call",
                phase="trajectory",
                category="trajectory",
                attrs=tool_attrs,
                traj_id=traj_id,
                sample_id=sample_id,
            )
            _tool_span.__enter__()
            try:
                observation, reward, terminated, truncated, info = self.env.step(action=response)
            finally:
                _tool_span.__exit__(None, None, None)
        else:
            observation, reward, terminated, truncated, info = self.env.step(action=response)

        self.rollout_cache.step += 1

        # terminated 完全由swe|tb env决定
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.stop_reason = EpisodeStopReason.MAX_STEPS
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['request_id'] = request_id
        self.rollout_cache.history[-1]['llm_response'] = response
        self.rollout_cache.history[-1]['terminated'] = terminated
        self.rollout_cache.history[-1]['truncated'] = truncated
        self.rollout_cache.history[-1]['step_kind'] = self._classify_step_kind(response, tool_names, terminated)
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append({
            "observation": copy.deepcopy(observation),
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None
        })
        return self.rollout_cache

    def make_decision(
        self,
        rollout_cache: RolloutCache,
        *,
        tracer=None,
        trace_context=None,
        trace_attrs: Optional[Dict[str, Any]] = None,
        sample_id: Optional[str] = None,
        traj_id: Optional[str] = None,
    ):
        tracer = tracer or get_trace_manager(component=f"env_{self.env_config['env_id']}")
        trace_attrs = dict(trace_attrs or {})
        lm_input = self.format_messages(rollout_cache)
        input_ids = lm_input.batch["input_ids"]

        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {input_ids.shape[1]},"
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        max_new_tokens = min(self.env_config["max_tokens_per_step"],
                             self.worker_config.generating_args.max_new_tokens,
                             self.pipeline_config.sequence_length-input_ids.shape[1])
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]
        lm_input.meta_info["trace_step"] = self.current_step
        request_id = uuid.uuid4().hex
        lm_input.meta_info["trace_request_id"] = request_id

        content = self.rollout_cache.history[-1]
        input_messages = content['observation']
        prompt_tokens = int(input_ids.shape[1])
        message_roles = [msg.get("role", "unknown") for msg in input_messages if isinstance(msg, dict)]
        last_message = input_messages[-1] if input_messages else {}
        request_attrs = {
            **trace_attrs,
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "message_count": len(input_messages) if input_messages else 0,
            "message_roles": message_roles,
            "last_message_role": last_message.get("role", "") if isinstance(last_message, dict) else "",
            "last_message_preview": self._normalize_preview(last_message.get("content", "")) if isinstance(last_message, dict) else "",
        }
        request_span = tracer.span(
            "inference.request",
            phase="inference",
            category="inference",
            trace_context=trace_context,
            sample_id=sample_id,
            traj_id=traj_id,
            attrs=request_attrs,
        )
        request_span.__enter__()

        output_tokens = 0
        response_preview = ""
        finish_reasons: list[str] = []
        phase_timing: Any = None
        try:
            lm_output: DataProto = self.llm_proxy.generate(
                messages=input_messages,
                lm_input=lm_input,
                generation_config=generation_config,
            )

            if lm_output is None:
                finish_reasons = ["abort"]
                request_span.update_attributes(output_tokens=0, finish_reasons=finish_reasons)
                return DataProto(
                    meta_info={
                        "stop_reason": GenerateStopReason.ABORT,
                        "request_id": request_id,
                        "trace_request_id": request_id,
                        "trace_sample_id": sample_id,
                        "trace_traj_id": traj_id,
                        "trace_env_step": trace_attrs.get("env_step"),
                    }
                )

            response_ids = lm_output.batch['responses'][0]
            response_ids = response_ids.tolist()
            output_tokens = len(response_ids)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            response_preview = self._normalize_preview(response_text)
            finish_reasons = list(lm_output.meta_info.get("finish_reasons") or ["stop"])
            request_span.update_attributes(
                output_tokens=output_tokens,
                finish_reasons=finish_reasons,
                response_preview=response_preview,
            )

            if "infer_logprobs" in lm_output.batch.keys():
                infer_logprobs = lm_output.batch['infer_logprobs'][0][-len(response_ids):]
                content["infer_logprobs"] = infer_logprobs.tolist()

            phase_timing = lm_output.meta_info.get("vllm_phase_timing")
            content["request_id"] = request_id
            content["finish_reasons"] = finish_reasons
            content["response_ids"] = response_ids
            content["messages"].append({"role": "assistant", "content": response_text})

            lm_output.meta_info["request_id"] = lm_output.meta_info.get("request_id") or request_id
            lm_output.meta_info["trace_request_id"] = request_id
            lm_output.meta_info["trace_sample_id"] = sample_id
            lm_output.meta_info["trace_traj_id"] = traj_id
            lm_output.meta_info["trace_env_step"] = trace_attrs.get("env_step")
            lm_output.meta_info["finish_reasons"] = finish_reasons
            lm_output.meta_info["prompt_tokens"] = prompt_tokens
            lm_output.meta_info["output_tokens"] = output_tokens
            lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
            return lm_output
        finally:
            request_span.__exit__(None, None, None)
            self._record_inference_phase_spans(
                tracer,
                request_span,
                trace_attrs=trace_attrs,
                sample_id=sample_id,
                traj_id=traj_id,
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                response_preview=response_preview,
                phase_timing=phase_timing,
            )

    def format_messages(self, rollout_cache: RolloutCache) -> DataProto:
        current_cache = rollout_cache.history[-1]

        messages: List[Dict] = current_cache["observation"]

        prompt_ids = self.tokenizer.apply_chat_template(convert_list_content_str(messages, parse_tool_call_parameter_to_dict=self.pipeline_config.parse_tool_call_parameter_to_dict),
                                                        tools=self.tools,
                                                        tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                                        return_dict=False)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        # Huggingface Transformers prefer position_ids to be 0-based.
        # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
        # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
        # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
        position_ids = attention_mask.cumsum(dim=-1) - 1
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])

        current_cache["prompt_ids"] = prompt_ids
        current_cache['state_hash'] = compute_object_hash(messages)
        current_cache['messages'] = messages
        return lm_input

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """
        Construct step-wise training samples from the collected trajectory.
        TODO: 相同前序合并优化
              样本构造方法：
                - 按messages构造response_id
                - 按response_id构造，纯step_wise用
        """
        last_observation = []
        if 'observation' in rollout_cache.history[-1]:
            last_observation = rollout_cache.history[-1]['observation']
            rollout_cache.history.pop(-1)

        samples: List[DataProto] = []
        step_rewards = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(step_rewards)

        # Initialize lists for step length statistics
        step_prompt_length_list = []
        step_response_length_list = []

        all_messages: List[List[Dict]] = [] # 可能包含多条轨迹，相同前序的为一条messages
        messages = None
        for step, history in enumerate(rollout_cache.history):
            if "response_ids" not in history:
                break

            # Collect step length statistics
            step_prompt_length_list.append(len(history["prompt_ids"]))
            step_response_length_list.append(len(history["response_ids"]))

            token_ids = history["prompt_ids"] + history["response_ids"]
            response_masks = [0] * len(history["prompt_ids"]) + [1] * len(history["response_ids"])
            input_ids =torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
            response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
            infer_logprobs = []
            if "infer_logprobs" in history:
                infer_logprobs = [0] * len(history["prompt_ids"]) + history["infer_logprobs"]

            generate_time = self.log_stats["generate_time"][len(self.log_stats["response_length"])]
            self.log_stats["response_length"].append(len(history["response_ids"]))
            if generate_time > 0.01:
                tokens_per_second = len(history["response_ids"]) / generate_time
                self.log_stats["tokens_per_second"].append(tokens_per_second)
            else:
                self.log_stats["tokens_per_second"].append(0.0)

            first_response_idx = response_masks.index(1)
            prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
            prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
            score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
            score_tensor[0][-1] = history['reward']
            # Huggingface Transformers prefer position_ids to be 0-based.
            # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
            # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
            # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
            position_ids = attention_mask.cumsum(dim=-1) - 1

            input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
            attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
            response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)
            lm_input = DataProto(
                batch=TensorDict(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "response_mask": response_mask,
                        "prompt_mask": prompt_mask,
                        "scores": score_tensor,
                    },
                    batch_size=input_ids.shape[0]),
                non_tensor_batch={
                    "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
                    "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
                    "tags": np.array([self.rollout_cache.tag], dtype=object),
                    "step_scores": np.array([history["reward"]], dtype=object), # step-level reward, return by env
                    "episode_scores": np.array([episode_score], dtype=object),
                    "state_hash": np.array([history['state_hash']], dtype=object),
                    "step": np.array([step], dtype=object),
                    "trajectory_data": np.array([None], dtype=object),
                    "messages": np.array([None], dtype=object),
                    "tools": np.array([None], dtype=object),
                    "exp_name": np.array([self.pipeline_config.exp_name], dtype=object),
                }
            )
            if len(infer_logprobs):
                infer_logprobs = torch.tensor(infer_logprobs, dtype=torch.float).unsqueeze(0)
                infer_logprobs = pad_to_length(infer_logprobs, length=self.pipeline_config.sequence_length, pad_value=0)
                lm_input.batch["infer_logprobs"] = infer_logprobs[:, 1:]

            samples.append(lm_input)
            messages = history["messages"]

        # TODO: 需要更细致的处理
        #       可选的方式是，将content + tool_use dict 替换回response
        all_messages.append(messages)
        batch: DataProto = DataProto.concat(samples)

        response_length = batch.batch["response_mask"].float().sum(-1).mean().item()
        metrics_agg_mode = self.rollout_cache.history[-1].get('metrics_agg_mode', {})
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        env_metric["num_actions"] = rollout_cache.step
        env_metric["env_timeout"] = getattr(self.env, "env_timeout", False)
        timing_metric = {
            "trajectory.total_time": round(float(time.time() - self.traj_start_time), 4),
            "trajectory.reset_time": round(float(self.log_stats["reset_time"]), 4),
            "trajectory.step_time.mean": round(float(np.mean(self.log_stats["step_time"])), 4),
            "trajectory.step_time.min": round(float(np.min(self.log_stats["step_time"])), 4),
            "trajectory.step_time.max": round(float(np.max(self.log_stats["step_time"])), 4),
            "trajectory.generation_time.mean": round(float(np.mean(self.log_stats["generate_time"])), 4),
            "trajectory.generation_time.min": round(float(np.min(self.log_stats["generate_time"])), 4),
            "trajectory.generation_time.max": round(float(np.max(self.log_stats["generate_time"])), 4),
            "trajectory.generation_time.total": round(float(np.sum(self.log_stats["generate_time"])), 4),
            "trajectory.throughput.mean": round(float(np.mean(self.log_stats["tokens_per_second"])), 4),
            "trajectory.throughput.min": round(float(np.min(self.log_stats["tokens_per_second"])), 4),
            "trajectory.throughput.max": round(float(np.max(self.log_stats["tokens_per_second"])), 4),
        }
        length_metric = {
            "trajectory.response_length": float(response_length),
            "trajectory.prompt_length.mean": round(float(np.mean(step_prompt_length_list)), 2),
            "trajectory.prompt_length.min": round(float(np.min(step_prompt_length_list)), 2),
            "trajectory.prompt_length.max": round(float(np.max(step_prompt_length_list)), 2),
            "trajectory.response_length_per_step.mean": round(float(np.mean(step_response_length_list)), 2),
            "trajectory.response_length_per_step.min": round(float(np.min(step_response_length_list)), 2),
            "trajectory.response_length_per_step.max": round(float(np.max(step_response_length_list)), 2),
        }

        env_metric.update(timing_metric)
        env_metric.update(length_metric)

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        batch.meta_info = {"metrics": env_metric}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        start_step = self.log_stats["current_step"][0]
        end_step = self.log_stats["current_step"][-1]
        last_step_info = rollout_cache.history[-1]
        failure_mode = last_step_info.get("failure_mode", "")
        traj_id = f"{rollout_cache.tag}_{start_step}_{end_step}_{rollout_cache.group_id}_{rollout_cache.env_id}_{self.episode_id}_{self.group_seed}_{timestamp}"
        trajectory_data = {
            "trajectory_id": traj_id,
            "timestamp": timestamp,
            "current_step": self.current_step,
            "env_info":{
                "env_id": rollout_cache.env_id,
                "group_id": rollout_cache.group_id,
                "tag": rollout_cache.tag,
                "seed": self.group_seed,
                "episode_id": self.episode_id,
                "max_steps": self.env_config.max_steps,
                "mode": self.mode,
                "sequence_length": self.pipeline_config.sequence_length,
                **self.env.env_info
            },
            "timing_info": {
                "traj_save_time": datetime.now().isoformat(),
                **timing_metric
            },
            "length_info": {
                "trajectory_length": rollout_cache.step,
                "num_actions": rollout_cache.step,
                "terminated": rollout_cache.terminated,
                "truncated": rollout_cache.truncated,
                **length_metric
            },
            "reward_info": {
                "episode_reward": episode_score,
                "step_rewards": step_rewards,
                "first_round_reward": step_rewards[0] if step_rewards else 0,
                "final_reward": step_rewards[-1] if step_rewards else 0
            },
            "failure_info": {
                "failure_mode": last_step_info.get("failure_mode", ""),
                "stop_reason": self.stop_reason.name,
                "error_messages": last_step_info.get("error_messages", []),
                "test_output": last_step_info.get("test_output", ""),
                "has_failure": bool(failure_mode and failure_mode not in ['', 'none']),
                "failure_step": rollout_cache.step,
            },
            "metrics": env_metric,
            "last_observation": last_observation
        }

        # stepwise 样本只存一份traj data
        batch.non_tensor_batch["trajectory_data"][-1] = json.dumps(trajectory_data)
        batch.non_tensor_batch["messages"][-1] = json.dumps(all_messages)
        batch.non_tensor_batch["tools"][-1] = json.dumps(self.tools)

        # 避免 trajectory_data dict 过大，导致写入/读取odps失败
        colummns_config = [
            ["trajectory_data", "string"],
            ["messages", "string"],
            ["tools", "string"],
            ["exp_name", "string"],
        ]
        batch.meta_info["COLUMMNS_CONFIG"] = colummns_config
        return batch

    def create_placeholder_rollout(self, episode_id):
        """
                Create a minimal placeholder rollout with response_mask=1 to skip loss calculation.
                """
        self.logger.info(f"[PLACEHOLDER_ROLLOUT] failure_mode: {self.failure_mode}")

        seq_len = length=self.pipeline_config.sequence_length
        input_ids = torch.full((1, seq_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
        position_ids = torch.zeros((1, seq_len), dtype=torch.long)
        response_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        prompt_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        score_tensor = torch.zeros((1, seq_len), dtype=torch.float)

        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        }, batch_size=1)


        infer_logprobs = torch.zeros((1, seq_len - 1), dtype=torch.float)
        lm_input.batch["infer_logprobs"] = infer_logprobs

        lm_input.non_tensor_batch = {
            "env_ids": np.array([self.env_config['env_id']], dtype=object),
            "group_ids": np.array([self.env_config['group_id']], dtype=object),
            "tags": np.array([self.env_config['tag']], dtype=object),
            "step_scores": np.array([0], dtype=object),
            "episode_scores": np.array([0], dtype=object),
            "state_hash": np.array([''], dtype=object),
            "step": np.array([0], dtype=object),
            "trajectory_data": np.array([None], dtype=object),
            "messages": np.array([None], dtype=object),
            "tools": np.array([None], dtype=object),
            "exp_name": np.array([self.pipeline_config.exp_name], dtype=object),
        }

        traj_group_id = f"{self.env_config['tag']}_{self.env_config['group_id']}_{episode_id}_{self.group_seed}"
        traj_id = f"{traj_group_id}_{self.env_config['env_id']}"
        lm_input.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * lm_input.batch.batch_size[0], dtype=object)
        lm_input.non_tensor_batch["traj_id"] = np.array([traj_id] * lm_input.batch.batch_size[0], dtype=object)

        colummns_config = [
            ["trajectory_data", "string"],
            ["messages", "string"],
            ["tools", "string"],
            ["exp_name", "string"],
        ]
        lm_input.meta_info["COLUMMNS_CONFIG"] = colummns_config
        lm_input.meta_info["metrics"] = {}
        return lm_input



class GroupFilter:
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        self.config = config
        self.env_manager_config = env_manager_config
        self.mode = mode
        self.global_filter_stats = {"total": 0, "filtered": 0}

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        self.global_filter_stats["total"] += 1
        should_drop = False
        for data in group:
            if data.meta_info.get("drop_flag", False):
                should_drop = True

        if not should_drop:
            return False

        current_global_filter_ratio = (
            self.global_filter_stats["filtered"] / self.global_filter_stats["total"]
            if self.global_filter_stats["total"] > 0 else 0.0
        )

        if current_global_filter_ratio >= 0.5:
            return False

        if (self.global_filter_stats["filtered"] + 1) / self.global_filter_stats["total"] > 0.5:
            return False

        self.global_filter_stats["filtered"] += 1
        return True
