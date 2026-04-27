#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

METRICS_LINE_RE = re.compile(r"\{.*\}")


def _iter_json_objects(log_path: Path):
    if not log_path.exists():
        return
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "timing.rollout" not in line and "throughput.total_samples" not in line:
                continue
            match = METRICS_LINE_RE.search(line)
            if not match:
                continue
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _load_last_metrics(log_path: Path) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for obj in _iter_json_objects(log_path):
        if "timing.rollout" in obj or "throughput.total_samples" in obj:
            last = obj
    return last


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _count_trace_files(trace_dir: Path) -> dict[str, int]:
    raw = trace_dir / "raw"
    return {
        "raw_misc_files": sum(1 for _ in (raw / "misc").glob("*.jsonl")) if (raw / "misc").exists() else 0,
        "raw_step_files": sum(1 for _ in (raw / "steps").rglob("*.jsonl")) if (raw / "steps").exists() else 0,
        "raw_sample_files": sum(1 for _ in (raw / "samples").rglob("*.jsonl")) if (raw / "samples").exists() else 0,
    }


def summarize_row(row: dict[str, str]) -> dict[str, Any]:
    log_path = Path(row["log_path"])
    trace_dir = Path(row["trace_dir"])
    metrics = _load_last_metrics(log_path)
    rollout_seconds = _safe_float(metrics.get("timing.rollout"))
    batch_size = int(row["batch_size"])
    traj_per_second = batch_size / rollout_seconds if rollout_seconds and rollout_seconds > 0 else None
    traj_per_hour = traj_per_second * 3600 if traj_per_second is not None else None

    output = {
        "experiment": row["experiment"],
        "batch_size": batch_size,
        "status": row["status"],
        "rollout_seconds": rollout_seconds,
        "traj_per_second": traj_per_second,
        "traj_per_hour": traj_per_hour,
        "throughput_total_samples": _safe_float(metrics.get("throughput.total_samples")),
        "score_mean": _safe_float(metrics.get("val/score/mean")),
        "log_path": str(log_path),
        "trace_dir": str(trace_dir),
    }
    output.update(_count_trace_files(trace_dir))

    for key in sorted(metrics):
        if key.startswith("vllm.") or "lmcache" in key.lower() or "kv" in key.lower():
            value = _safe_float(metrics[key])
            if value is not None:
                output[key] = value
    return output


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], output_md: Path) -> None:
    columns = [
        "experiment",
        "batch_size",
        "status",
        "rollout_seconds",
        "traj_per_second",
        "traj_per_hour",
        "raw_misc_files",
        "raw_step_files",
        "raw_sample_files",
        "log_path",
    ]
    lines = ["# Batch Scale Active-Only HBM Summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_number(row.get(column)) for column in columns) + " |")
    lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = [summarize_row(row) for row in manifest_rows]
    rows.sort(key=lambda row: (row["experiment"], row["batch_size"]))
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
