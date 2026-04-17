import argparse
import json
import shutil
import time
from pathlib import Path

from roll.utils.tracing import NullTraceManager, TraceManager, TracingConfig, export_trace_directory


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _payload_work(units: int) -> int:
    acc = 0
    for idx in range(units):
        acc = (acc * 1315423911 + idx) % 1_000_003
    return acc


def generate_demo_trace(trace_dir: Path) -> list[str]:
    config = TracingConfig(
        enable=True,
        output_dir=str(trace_dir),
        export_html=True,
        export_step_interval=1,
        flush_every_n_spans=256,
    )
    manager = TraceManager(config, component="trace_demo")

    for step in range(2):
        with manager.span(
            "pipeline.step",
            phase="pipeline",
            step=step,
            attrs={"mode": "demo", "step_index": step},
        ) as step_span:
            with manager.span(
                "scheduler.suspend",
                phase="scheduler",
                category="scheduler",
                trace_context=step_span.child_context(),
            ):
                _sleep(0.002)

            for traj_idx in range(2):
                traj_ctx = step_span.child_context(
                    sample_id=f"sample-{step}-{traj_idx}",
                    traj_id=f"traj-{step}-{traj_idx}",
                    tags={"trajectory_index": traj_idx},
                )
                with manager.span(
                    "inference.generate",
                    phase="inference",
                    category="inference",
                    trace_context=traj_ctx,
                ) as generation_span:
                    with manager.span(
                        "inference.prefill",
                        phase="inference",
                        category="inference",
                        trace_context=generation_span.child_context(),
                    ):
                        _sleep(0.004)
                    with manager.span(
                        "inference.decode",
                        phase="inference",
                        category="inference",
                        trace_context=generation_span.child_context(),
                    ):
                        _sleep(0.006)
                    with manager.span(
                        "inference.overhead",
                        phase="inference",
                        category="inference",
                        trace_context=generation_span.child_context(),
                    ):
                        _sleep(0.001)
                    if traj_idx == 0:
                        with manager.span(
                            "trajectory.tool_call",
                            phase="trajectory",
                            category="trajectory",
                            trace_context=generation_span.child_context(),
                            attrs={"tool_name": "search_code"},
                        ):
                            _sleep(0.003)

                with manager.span(
                    "reward.compute",
                    phase="reward",
                    category="reward",
                    trace_context=traj_ctx,
                ):
                    _sleep(0.0015)

            with manager.span(
                "policy_eval.logprob",
                phase="policy_eval",
                category="policy_eval",
                trace_context=step_span.child_context(),
            ):
                _sleep(0.001)
            with manager.span(
                "advantage.compute",
                phase="advantage",
                category="advantage",
                trace_context=step_span.child_context(),
            ):
                _sleep(0.001)
            with manager.span(
                "training.actor_update",
                phase="training",
                category="training",
                trace_context=step_span.child_context(),
            ):
                _sleep(0.003)
            with manager.span(
                "weight_sync.sync_weights",
                phase="weight_sync",
                category="weight_sync",
                trace_context=step_span.child_context(),
            ):
                _sleep(0.002)

        manager.flush(step=step)
        manager.maybe_export_step(step)

    return export_trace_directory(str(trace_dir))


def _run_benchmark_workload(manager, *, steps: int, trajectories: int, payload_units: int) -> int:
    total_spans = 0
    payload_sink = 0
    for step in range(steps):
        with manager.span("pipeline.step", phase="pipeline", step=step) as step_span:
            total_spans += 1
            for traj_idx in range(trajectories):
                traj_ctx = step_span.child_context(
                    sample_id=f"sample-{step}-{traj_idx}",
                    traj_id=f"traj-{step}-{traj_idx}",
                )
                with manager.span("inference.generate", phase="inference", trace_context=traj_ctx) as generation_span:
                    total_spans += 1
                    with manager.span(
                        "inference.prefill",
                        phase="inference",
                        trace_context=generation_span.child_context(),
                    ):
                        payload_sink ^= _payload_work(payload_units)
                        total_spans += 1
                    with manager.span(
                        "inference.decode",
                        phase="inference",
                        trace_context=generation_span.child_context(),
                    ):
                        payload_sink ^= _payload_work(payload_units)
                        total_spans += 1
                    manager.record_completed_span(
                        "inference.overhead",
                        phase="inference",
                        trace_context=generation_span.child_context(),
                        start_time_ns=time.time_ns(),
                        end_time_ns=time.time_ns(),
                    )
                    total_spans += 1
                manager.record_completed_span(
                    "rollout.get_batch",
                    phase="rollout",
                    category="rollout",
                    trace_context=traj_ctx,
                    start_time_ns=time.time_ns(),
                    end_time_ns=time.time_ns(),
                )
                payload_sink ^= _payload_work(payload_units)
                total_spans += 1
            with manager.span(
                "reward.compute",
                phase="reward",
                trace_context=step_span.child_context(),
            ):
                payload_sink ^= _payload_work(payload_units)
                total_spans += 1
            with manager.span(
                "advantage.compute",
                phase="advantage",
                trace_context=step_span.child_context(),
            ):
                payload_sink ^= _payload_work(payload_units)
                total_spans += 1
            with manager.span(
                "training.actor_update",
                phase="training",
                trace_context=step_span.child_context(),
            ):
                payload_sink ^= _payload_work(payload_units)
                total_spans += 1
    manager.flush()
    if payload_sink == -1:
        raise AssertionError("unreachable")
    return total_spans


def run_benchmark(trace_dir: Path, *, steps: int, trajectories: int, payload_units: int) -> dict:
    baseline_start = time.perf_counter()
    payload_sink = 0
    baseline_operations = 0
    for step in range(steps):
        for traj_idx in range(trajectories):
            payload_sink ^= _payload_work(payload_units)
            payload_sink ^= _payload_work(payload_units)
            payload_sink ^= _payload_work(payload_units)
            baseline_operations += 5
            _ = (step, traj_idx)
        payload_sink ^= _payload_work(payload_units)
        payload_sink ^= _payload_work(payload_units)
        payload_sink ^= _payload_work(payload_units)
        baseline_operations += 4
    baseline_s = time.perf_counter() - baseline_start
    if payload_sink == -1:
        raise AssertionError("unreachable")

    disabled_manager = NullTraceManager()
    disabled_start = time.perf_counter()
    disabled_total_spans = _run_benchmark_workload(
        disabled_manager,
        steps=steps,
        trajectories=trajectories,
        payload_units=payload_units,
    )
    disabled_s = time.perf_counter() - disabled_start

    config = TracingConfig(
        enable=True,
        output_dir=str(trace_dir / "benchmark_trace"),
        export_html=False,
        flush_every_n_spans=max(steps * trajectories * 8, 1024),
    )
    enabled_manager = TraceManager(config, component="trace_benchmark")
    enabled_start = time.perf_counter()
    enabled_total_spans = _run_benchmark_workload(
        enabled_manager,
        steps=steps,
        trajectories=trajectories,
        payload_units=payload_units,
    )
    enabled_s = time.perf_counter() - enabled_start

    assert disabled_total_spans == enabled_total_spans

    per_step_baseline_ms = baseline_s * 1000 / max(steps, 1)
    per_step_enabled_ms = enabled_s * 1000 / max(steps, 1)
    overhead_pct = ((enabled_s - baseline_s) / baseline_s * 100) if baseline_s > 0 else 0.0
    overhead_pct_vs_disabled = ((enabled_s - disabled_s) / disabled_s * 100) if disabled_s > 0 else 0.0
    per_span_overhead_us = ((enabled_s - disabled_s) / max(enabled_total_spans, 1)) * 1_000_000

    report = {
        "steps": steps,
        "trajectories_per_step": trajectories,
        "payload_units": payload_units,
        "baseline_operations": baseline_operations,
        "total_spans": enabled_total_spans,
        "baseline_seconds": round(baseline_s, 6),
        "disabled_api_seconds": round(disabled_s, 6),
        "enabled_seconds": round(enabled_s, 6),
        "baseline_ms_per_step": round(per_step_baseline_ms, 6),
        "enabled_ms_per_step": round(per_step_enabled_ms, 6),
        "enabled_overhead_pct_vs_baseline": round(overhead_pct, 4),
        "enabled_overhead_pct_vs_disabled_api": round(overhead_pct_vs_disabled, 4),
        "enabled_overhead_us_per_span_vs_disabled_api": round(per_span_overhead_us, 6),
        "notes": [
            "Synthetic microbenchmark; measures tracing API cost in-process only.",
            "Does not include distributed worker execution, GPU kernels, or HTML export time.",
        ],
    }
    return report


def write_benchmark_report(report: dict, output_path: Path) -> None:
    md = "\n".join(
        [
            "# ROLL Tracing Benchmark Report",
            "",
            f"- steps: `{report['steps']}`",
            f"- trajectories_per_step: `{report['trajectories_per_step']}`",
            f"- payload_units: `{report['payload_units']}`",
            f"- total_spans: `{report['total_spans']}`",
            f"- baseline_seconds: `{report['baseline_seconds']}`",
            f"- disabled_api_seconds: `{report['disabled_api_seconds']}`",
            f"- enabled_seconds: `{report['enabled_seconds']}`",
            f"- baseline_ms_per_step: `{report['baseline_ms_per_step']}`",
            f"- enabled_ms_per_step: `{report['enabled_ms_per_step']}`",
            f"- enabled_overhead_pct_vs_baseline: `{report['enabled_overhead_pct_vs_baseline']}`",
            f"- enabled_overhead_pct_vs_disabled_api: `{report['enabled_overhead_pct_vs_disabled_api']}`",
            f"- enabled_overhead_us_per_span_vs_disabled_api: `{report['enabled_overhead_us_per_span_vs_disabled_api']}`",
            "",
            "Notes:",
            *[f"- {note}" for note in report["notes"]],
        ]
    )
    output_path.write_text(md + "\n", encoding="utf-8")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Generate demo trace timelines and tracing benchmark artifacts.")
    parser.add_argument("--output-dir", default="./output/tracing_artifacts", help="Directory for generated artifacts.")
    parser.add_argument(
        "--docs-static-dir",
        default=None,
        help="Optional docs static directory to receive a copy of the generated artifacts.",
    )
    parser.add_argument("--benchmark-steps", type=int, default=80)
    parser.add_argument("--benchmark-trajectories", type=int, default=8)
    parser.add_argument("--benchmark-payload-units", type=int, default=8192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    trace_dir = output_dir / "demo_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    html_outputs = generate_demo_trace(trace_dir)
    benchmark_report = run_benchmark(
        output_dir,
        steps=args.benchmark_steps,
        trajectories=args.benchmark_trajectories,
        payload_units=args.benchmark_payload_units,
    )

    benchmark_json_path = output_dir / "benchmark_report.json"
    benchmark_md_path = output_dir / "benchmark_report.md"
    benchmark_json_path.write_text(json.dumps(benchmark_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_benchmark_report(benchmark_report, benchmark_md_path)

    manifest = {
        "trace_dir": str(trace_dir),
        "timeline_index": str(trace_dir / "timeline" / "index.html"),
        "step_timelines": html_outputs,
        "benchmark_report_json": str(benchmark_json_path),
        "benchmark_report_md": str(benchmark_md_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.docs_static_dir:
        copy_tree(output_dir, Path(args.docs_static_dir).resolve())

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
