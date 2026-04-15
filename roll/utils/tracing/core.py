import contextvars
import dataclasses
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from roll.utils.logging import get_logger

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - optional dependency at import time
    DictConfig = None
    OmegaConf = None


logger = get_logger()

_CURRENT_TRACE_CONTEXT: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar(
    "roll_trace_context",
    default=None,
)
_TRACE_MANAGERS: dict[tuple[str, int], "TraceManager"] = {}
_TRACE_MANAGER_LOCK = threading.Lock()


def _coerce_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {k: _coerce_value(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _coerce_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _sanitize_filename(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", value or "trace")
    return text.strip("._") or "trace"


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class TracingConfig:
    enable: bool = field(default=False, metadata={"help": "Enable structured tracing."})
    output_dir: Optional[str] = field(default=None, metadata={"help": "Trace artifact output directory."})
    export_html: bool = field(default=True, metadata={"help": "Export interactive HTML timelines."})
    export_step_interval: int = field(
        default=1,
        metadata={"help": "Export one timeline page every N traced training steps."},
    )
    flush_every_n_spans: int = field(
        default=32,
        metadata={"help": "Flush buffered spans to disk every N completed spans per process."},
    )
    capture_rollout_samples: bool = field(
        default=True,
        metadata={"help": "Capture sample- and trajectory-level rollout spans."},
    )
    capture_queue_management: bool = field(
        default=True,
        metadata={"help": "Capture queue manager spans and queue-side phase events."},
    )
    capture_generation_subphases: bool = field(
        default=True,
        metadata={"help": "Capture generation subphases such as prefill/decode/overhead."},
    )
    max_exported_spans_per_step: int = field(
        default=20000,
        metadata={"help": "Soft limit for exported spans per step HTML page."},
    )

    def resolve_output_dir(self, base_output_dir: Optional[str] = None) -> str:
        if self.output_dir:
            return self.output_dir
        if base_output_dir:
            return os.path.join(base_output_dir, "traces")
        return os.environ.get("ROLL_TRACE_DIR", "./output/traces")

    def apply_env(self, base_output_dir: Optional[str] = None) -> None:
        os.environ["ROLL_TRACE_ENABLE"] = "1" if self.enable else "0"
        os.environ["ROLL_TRACE_DIR"] = self.resolve_output_dir(base_output_dir=base_output_dir)
        os.environ["ROLL_TRACE_EXPORT_HTML"] = "1" if self.export_html else "0"
        os.environ["ROLL_TRACE_EXPORT_STEP_INTERVAL"] = str(self.export_step_interval)
        os.environ["ROLL_TRACE_FLUSH_EVERY_N_SPANS"] = str(self.flush_every_n_spans)
        os.environ["ROLL_TRACE_CAPTURE_ROLLOUT_SAMPLES"] = "1" if self.capture_rollout_samples else "0"
        os.environ["ROLL_TRACE_CAPTURE_QUEUE"] = "1" if self.capture_queue_management else "0"
        os.environ["ROLL_TRACE_CAPTURE_GENERATION"] = "1" if self.capture_generation_subphases else "0"
        os.environ["ROLL_TRACE_MAX_EXPORTED_SPANS_PER_STEP"] = str(self.max_exported_spans_per_step)

    @classmethod
    def from_env(cls) -> "TracingConfig":
        return cls(
            enable=_bool_env("ROLL_TRACE_ENABLE", False),
            output_dir=os.environ.get("ROLL_TRACE_DIR"),
            export_html=_bool_env("ROLL_TRACE_EXPORT_HTML", True),
            export_step_interval=_int_env("ROLL_TRACE_EXPORT_STEP_INTERVAL", 1),
            flush_every_n_spans=_int_env("ROLL_TRACE_FLUSH_EVERY_N_SPANS", 32),
            capture_rollout_samples=_bool_env("ROLL_TRACE_CAPTURE_ROLLOUT_SAMPLES", True),
            capture_queue_management=_bool_env("ROLL_TRACE_CAPTURE_QUEUE", True),
            capture_generation_subphases=_bool_env("ROLL_TRACE_CAPTURE_GENERATION", True),
            max_exported_spans_per_step=_int_env("ROLL_TRACE_MAX_EXPORTED_SPANS_PER_STEP", 20000),
        )


@dataclass
class TraceContext:
    trace_id: str
    parent_id: Optional[str] = None
    step: Optional[int] = None
    sample_id: Optional[str] = None
    traj_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    def child(self, **kwargs) -> "TraceContext":
        return TraceContext(
            trace_id=kwargs.pop("trace_id", self.trace_id),
            parent_id=kwargs.pop("parent_id", self.parent_id),
            step=kwargs.pop("step", self.step),
            sample_id=kwargs.pop("sample_id", self.sample_id),
            traj_id=kwargs.pop("traj_id", self.traj_id),
            tags={**self.tags, **_coerce_value(kwargs.pop("tags", {}))},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "step": self.step,
            "sample_id": self.sample_id,
            "traj_id": self.traj_id,
            "tags": _coerce_value(self.tags),
        }

    @classmethod
    def from_any(cls, value: Any) -> Optional["TraceContext"]:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                trace_id=value.get("trace_id") or uuid.uuid4().hex,
                parent_id=value.get("parent_id"),
                step=value.get("step"),
                sample_id=value.get("sample_id"),
                traj_id=value.get("traj_id"),
                tags=_coerce_value(value.get("tags", {})),
            )
        raise TypeError(f"Unsupported trace context type: {type(value)}")


class NullTraceSpan:
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def update_attributes(self, **attrs) -> None:
        return None

    def child_context(self, **kwargs) -> Dict[str, Any]:
        return {}

    def finish(self, flush: bool = False) -> None:
        return None


class TraceSpan:
    def __init__(
        self,
        manager: "TraceManager",
        name: str,
        *,
        phase: str,
        category: Optional[str] = None,
        trace_context: Any = None,
        step: Optional[int] = None,
        sample_id: Optional[str] = None,
        traj_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        start_time_ns: Optional[int] = None,
    ):
        context = TraceContext.from_any(trace_context) or _CURRENT_TRACE_CONTEXT.get()
        inherited_tags = {} if context is None else dict(context.tags)
        effective_step = step if step is not None else (None if context is None else context.step)
        effective_sample_id = sample_id if sample_id is not None else (None if context is None else context.sample_id)
        effective_traj_id = traj_id if traj_id is not None else (None if context is None else context.traj_id)
        effective_trace_id = trace_id or (None if context is None else context.trace_id) or uuid.uuid4().hex
        effective_parent_id = parent_id if parent_id is not None else (None if context is None else context.parent_id)

        self.manager = manager
        self.name = name
        self.phase = phase
        self.category = category or phase
        self.trace_id = effective_trace_id
        self.parent_id = effective_parent_id
        self.step = effective_step
        self.sample_id = effective_sample_id
        self.traj_id = effective_traj_id
        self.span_id = uuid.uuid4().hex
        self.attrs = {**inherited_tags, **_coerce_value(attrs or {})}
        self.start_wall_ns = start_time_ns or time.time_ns()
        self.start_perf_ns = time.perf_counter_ns()
        self.end_wall_ns: Optional[int] = None
        self.duration_ms: Optional[float] = None
        self._token = None
        self._finished = False

    def __enter__(self):
        self._token = _CURRENT_TRACE_CONTEXT.set(TraceContext(
            trace_id=self.trace_id,
            parent_id=self.span_id,
            step=self.step,
            sample_id=self.sample_id,
            traj_id=self.traj_id,
            tags=dict(self.attrs),
        ))
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.attrs["error"] = str(exc)
        if self._token is not None:
            _CURRENT_TRACE_CONTEXT.reset(self._token)
        self.finish()
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = _coerce_value(value)

    def update_attributes(self, **attrs) -> None:
        for key, value in attrs.items():
            self.set_attribute(key, value)

    def child_context(self, **kwargs) -> Dict[str, Any]:
        tags = {**self.attrs, **_coerce_value(kwargs.pop("tags", {}))}
        return TraceContext(
            trace_id=kwargs.pop("trace_id", self.trace_id),
            parent_id=kwargs.pop("parent_id", self.span_id),
            step=kwargs.pop("step", self.step),
            sample_id=kwargs.pop("sample_id", self.sample_id),
            traj_id=kwargs.pop("traj_id", self.traj_id),
            tags=tags,
        ).to_dict()

    def finish(self, flush: bool = False) -> None:
        if self._finished:
            return
        self._finished = True
        self.end_wall_ns = time.time_ns()
        self.duration_ms = round((time.perf_counter_ns() - self.start_perf_ns) / 1_000_000, 6)
        self.manager._record_completed(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_id=self.parent_id,
            name=self.name,
            phase=self.phase,
            category=self.category,
            step=self.step,
            sample_id=self.sample_id,
            traj_id=self.traj_id,
            attrs=self.attrs,
            start_time_ns=self.start_wall_ns,
            end_time_ns=self.end_wall_ns,
            duration_ms=self.duration_ms,
        )
        if flush:
            self.manager.flush(step=self.step)


class NullTraceManager:
    enabled = False

    def span(self, *args, **kwargs) -> NullTraceSpan:
        return NullTraceSpan()

    def start_span(self, *args, **kwargs) -> NullTraceSpan:
        return NullTraceSpan()

    def new_context(self, **kwargs) -> Dict[str, Any]:
        return {}

    def record_completed_span(self, *args, **kwargs) -> None:
        return None

    def flush(self, step: Optional[int] = None) -> None:
        return None

    def maybe_export_step(self, step: int) -> Optional[str]:
        return None


class TraceManager:
    def __init__(self, config: TracingConfig, base_output_dir: Optional[str] = None, component: Optional[str] = None):
        self.config = config
        self.enabled = bool(config.enable)
        self.output_dir = Path(config.resolve_output_dir(base_output_dir=base_output_dir))
        process_name = component or os.environ.get("WORKER_NAME") or os.environ.get("roll_EXEC_FUNC_NAME") or "process"
        self.process_label = _sanitize_filename(process_name)
        self.process_file_name = f"{self.process_label}-pid{os.getpid()}.jsonl"
        self._lock = threading.Lock()
        self._buffer: dict[str, list[dict[str, Any]]] = {}
        self._buffer_count = 0
        if self.enabled:
            (self.output_dir / "raw" / "steps").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "raw" / "misc").mkdir(parents=True, exist_ok=True)

    def _bucket_key(self, step: Optional[int]) -> str:
        if step is None:
            return "misc"
        return f"step_{int(step):06d}"

    def _bucket_path(self, step: Optional[int]) -> Path:
        if step is None:
            return self.output_dir / "raw" / "misc" / self.process_file_name
        return self.output_dir / "raw" / "steps" / self._bucket_key(step) / self.process_file_name

    def new_context(
        self,
        *,
        step: Optional[int] = None,
        sample_id: Optional[str] = None,
        traj_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return TraceContext(
            trace_id=trace_id or uuid.uuid4().hex,
            parent_id=parent_id,
            step=step,
            sample_id=sample_id,
            traj_id=traj_id,
            tags=_coerce_value(tags or {}),
        ).to_dict()

    def span(self, name: str, *, phase: str, category: Optional[str] = None, trace_context: Any = None,
             step: Optional[int] = None, sample_id: Optional[str] = None, traj_id: Optional[str] = None,
             parent_id: Optional[str] = None, trace_id: Optional[str] = None, attrs: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return NullTraceSpan()
        return TraceSpan(
            self,
            name,
            phase=phase,
            category=category,
            trace_context=trace_context,
            step=step,
            sample_id=sample_id,
            traj_id=traj_id,
            parent_id=parent_id,
            trace_id=trace_id,
            attrs=attrs,
        )

    def start_span(self, name: str, **kwargs):
        return self.span(name, **kwargs)

    def _record_completed(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_id: Optional[str],
        name: str,
        phase: str,
        category: str,
        step: Optional[int],
        sample_id: Optional[str],
        traj_id: Optional[str],
        attrs: Dict[str, Any],
        start_time_ns: int,
        end_time_ns: int,
        duration_ms: float,
    ) -> None:
        record = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": parent_id,
            "name": name,
            "phase": phase,
            "category": category,
            "step": step,
            "sample_id": sample_id,
            "traj_id": traj_id,
            "attrs": _coerce_value(attrs),
            "start_time_ns": int(start_time_ns),
            "end_time_ns": int(end_time_ns),
            "duration_ms": float(duration_ms),
            "pid": os.getpid(),
            "process_label": self.process_label,
        }
        bucket = self._bucket_key(step)
        should_flush = False
        with self._lock:
            self._buffer.setdefault(bucket, []).append(record)
            self._buffer_count += 1
            if self._buffer_count >= self.config.flush_every_n_spans:
                should_flush = True
        if should_flush:
            self.flush()

    def record_completed_span(
        self,
        name: str,
        *,
        phase: str,
        category: Optional[str] = None,
        trace_context: Any = None,
        step: Optional[int] = None,
        sample_id: Optional[str] = None,
        traj_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        start_time_ns: int,
        end_time_ns: int,
    ) -> None:
        if not self.enabled:
            return
        context = TraceContext.from_any(trace_context)
        self._record_completed(
            trace_id=trace_id or (None if context is None else context.trace_id) or uuid.uuid4().hex,
            span_id=uuid.uuid4().hex,
            parent_id=parent_id if parent_id is not None else (None if context is None else context.parent_id),
            name=name,
            phase=phase,
            category=category or phase,
            step=step if step is not None else (None if context is None else context.step),
            sample_id=sample_id if sample_id is not None else (None if context is None else context.sample_id),
            traj_id=traj_id if traj_id is not None else (None if context is None else context.traj_id),
            attrs=_coerce_value(attrs or {}),
            start_time_ns=int(start_time_ns),
            end_time_ns=int(end_time_ns),
            duration_ms=round((int(end_time_ns) - int(start_time_ns)) / 1_000_000, 6),
        )

    def flush(self, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            keys = [self._bucket_key(step)] if step is not None else list(self._buffer.keys())
            for key in keys:
                records = self._buffer.pop(key, [])
                if not records:
                    continue
                bucket_step = None if key == "misc" else int(key.split("_")[-1])
                path = self._bucket_path(bucket_step)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fp:
                    for record in records:
                        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                        self._buffer_count -= 1

    def maybe_export_step(self, step: int) -> Optional[str]:
        if not self.enabled or not self.config.export_html:
            return None
        if step % max(self.config.export_step_interval, 1) != 0:
            return None
        self.flush(step=step)
        from roll.utils.tracing.exporter import export_trace_step

        return export_trace_step(
            trace_dir=str(self.output_dir),
            step=step,
            max_exported_spans=self.config.max_exported_spans_per_step,
        )


def _resolve_trace_config(config: Any) -> tuple[TracingConfig, Optional[str]]:
    if isinstance(config, TracingConfig):
        return config, None
    if config is not None and hasattr(config, "tracing"):
        trace_config = getattr(config, "tracing")
        if isinstance(trace_config, TracingConfig):
            return trace_config, getattr(config, "output_dir", None)
        if DictConfig is not None and isinstance(trace_config, DictConfig):
            resolved = OmegaConf.to_container(trace_config, resolve=True) if OmegaConf is not None else dict(trace_config)
            return TracingConfig(**dict(resolved)), getattr(config, "output_dir", None)
        if isinstance(trace_config, Mapping):
            return TracingConfig(**dict(trace_config)), getattr(config, "output_dir", None)
    return TracingConfig.from_env(), None


def get_trace_manager(config: Any = None, component: Optional[str] = None) -> TraceManager | NullTraceManager:
    trace_config, base_output_dir = _resolve_trace_config(config)
    if not trace_config.enable:
        return NullTraceManager()

    output_dir = trace_config.resolve_output_dir(base_output_dir=base_output_dir)
    key = (os.path.abspath(output_dir), os.getpid())
    with _TRACE_MANAGER_LOCK:
        manager = _TRACE_MANAGERS.get(key)
        if manager is None:
            manager = TraceManager(trace_config, base_output_dir=base_output_dir, component=component)
            _TRACE_MANAGERS[key] = manager
        return manager
