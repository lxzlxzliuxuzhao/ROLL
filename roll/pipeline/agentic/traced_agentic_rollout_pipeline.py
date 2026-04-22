import json
import os.path
import time
from itertools import count

import ray
import torch
from codetiming import Timer

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import get_episode_scores
from roll.pipeline.agentic.agentic_rollout_pipeline import AgenticRolloutPipeline
from roll.pipeline.agentic.utils import dump_rollout_trajectories
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.tracing import get_trace_manager

logger = get_logger()


class TracedAgenticRolloutPipeline(AgenticRolloutPipeline):
    """Rollout-only pipeline with driver-side tracing and step export."""

    def _flush_remote_traces(self, step: int) -> None:
        remote_flush_refs = [self.rollout_scheduler.flush_traces.remote(step)]
        remote_flush_refs.extend(self.actor_infer.execute_all_async("flush_traces", step))
        ray.get(remote_flush_refs)

    @torch.no_grad()
    def run(self):
        tracer = get_trace_manager(config=self.pipeline_config, component="driver")

        for global_step in (count() if self.pipeline_config.max_steps == -1 else range(self.pipeline_config.max_steps)):
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}
            batch: DataProto = DataProto()
            batch.meta_info = {"global_step": global_step}
            should_stop = False

            step_span = tracer.span(
                "pipeline.step",
                phase="pipeline",
                step=global_step,
                attrs={"global_step": global_step},
            )
            step_span.__enter__()
            try:
                with Timer(name="rollout", logger=None) as rollout_timer:
                    load_span = tracer.span("weight_sync.load_infer", phase="weight_sync", step=global_step)
                    load_span.__enter__()
                    self.actor_infer.load_states()
                    load_span.__exit__(None, None, None)

                    rollout_span = tracer.span("rollout.collect_batch", phase="rollout", step=global_step)
                    rollout_span.__enter__()
                    get_batch_span = tracer.span("rollout.get_batch", phase="rollout", step=global_step)
                    get_batch_span.__enter__()
                    batch = ray.get(self.rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                    get_batch_span.__exit__(None, None, None)
                    if batch is None:
                        rollout_span.__exit__(None, None, None)
                        should_stop = True
                    else:
                        if "get_batch_return_start_time" in batch.meta_info:
                            metrics["timing.rollout.get_batch"] = time.time() - batch.meta_info.pop("get_batch_return_start_time")
                        actor_infer_metrics: DataProto = self.actor_infer.get_metrics()
                        metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))
                        rollout_span.__exit__(None, None, None)

                metrics["timing.rollout"] = rollout_timer.last
                if not should_stop:
                    eval_metrics = reduce_metrics(batch.meta_info.get("metrics", {}))
                    eval_score = get_episode_scores(batch)
                    eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
                    eval_metrics["score/max"] = torch.max(eval_score).detach().item()
                    eval_metrics["score/min"] = torch.min(eval_score).detach().item()

                    batch_grouped = batch.group_by(keys="tags")
                    for group_name, group_batch in batch_grouped.items():
                        eval_score = get_episode_scores(group_batch)
                        eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
                        eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
                        eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()
                        group_eval_metrics = reduce_metrics(group_batch.meta_info.get("metrics", {}))
                        eval_metrics.update({f"{group_name}/{k}": v for k, v in group_eval_metrics.items()})

                    metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
                    batch.meta_info["global_step"] = global_step
                    metrics["throughput.total_samples"] = (global_step + 1) * batch.batch.shape[0]

                    dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)
                    self.tracker.log(values=metrics, step=global_step)

                    if global_step % self.pipeline_config.logging_steps == 0:
                        if int(os.environ.get("RAY_PROFILING", "0")):
                            timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                            os.makedirs(timeline_dir, exist_ok=True)
                            ray.timeline(
                                filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                            )

                        prompt_mask = batch.batch["prompt_mask"]
                        non_prompt_mask = torch.logical_not(batch.batch["prompt_mask"])
                        input_ids = batch.batch["input_ids"]
                        prompt_ids = torch.where(
                            prompt_mask.bool(), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id)
                        )
                        response_ids = torch.where(
                            non_prompt_mask.bool(), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id)
                        )

                        generate_res = []
                        prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                        episode_scores = batch.non_tensor_batch["episode_scores"].tolist()
                        for prompt, response, episode_score in zip(prompts, responses, episode_scores):
                            generate_res.append(
                                {
                                    "prompt": prompt,
                                    "response": response,
                                    "episode_score": episode_score,
                                }
                            )
                        logger.info(json.dumps(generate_res[:10], ensure_ascii=False))
                        logger.info(json.dumps(metrics, ensure_ascii=False))

                    logger.info(f"pipeline step {global_step} finished")
            finally:
                step_span.__exit__(None, None, None)
                tracer.flush()
                self._flush_remote_traces(global_step)
                tracer.maybe_export_step(global_step)

            if should_stop:
                break

        ray.get(self.rollout_scheduler.shutdown.remote())
        logger.info("pipeline complete!")
