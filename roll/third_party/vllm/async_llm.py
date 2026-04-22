import asyncio

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils import cancel_task_threadsafe
from vllm.v1.engine.async_llm import AsyncLLM

from roll.third_party.vllm.request_timing import TracedOutputProcessor


logger = init_logger(__name__)


class CustomAsyncLLM(AsyncLLM):
    @staticmethod
    def _normalize_engine_snapshots(raw_snapshot) -> list[dict]:
        if raw_snapshot is None:
            return []
        if isinstance(raw_snapshot, dict):
            return [raw_snapshot]
        if isinstance(raw_snapshot, (list, tuple)):
            return [item for item in raw_snapshot if isinstance(item, dict)]
        return []

    def _ensure_traced_output_processor(self):
        if isinstance(self.output_processor, TracedOutputProcessor):
            return

        previous_output_processor = self.output_processor
        traced_output_processor = TracedOutputProcessor(self.tokenizer, log_stats=self.log_stats)
        traced_output_processor.tracer = previous_output_processor.tracer
        self.output_processor = traced_output_processor

    def _run_output_handler(self):
        self._ensure_traced_output_processor()
        super()._run_output_handler()
        self._run_stats_handler()

    def _run_stats_handler(self):
        if not self.log_stats or self.logger_manager is None:
            return

        stats_handler = getattr(self, "stats_handler", None)
        if stats_handler is not None and not stats_handler.done():
            return

        output_handler = self.output_handler
        if output_handler is None:
            return

        logger_manager = self.logger_manager
        stats_interval = envs.VLLM_LOG_STATS_INTERVAL

        async def stats_handler():
            try:
                while not output_handler.done():
                    await asyncio.sleep(stats_interval)
                    if output_handler.done():
                        break
                    await self._refresh_stats_logger_snapshots()
                    logger_manager.log()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("CustomAsyncLLM stats_handler failed.")

        self.stats_handler = asyncio.create_task(stats_handler())

    def shutdown(self):
        cancel_task_threadsafe(getattr(self, "stats_handler", None))
        super().shutdown()

    async def custom_init_worker(self):
        await self.engine_core.collective_rpc_async(method="custom_init_worker")

    async def load_states(self):
        await self.engine_core.collective_rpc_async(method="load_states")

    async def offload_states(self, level):
        await self.reset_prefix_cache()
        await self.engine_core.collective_rpc_async(method="offload_states", args=(level,))

    async def setup_collective_group(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="setup_collective_group", args=args, kwargs=kwargs)

    async def broadcast_parameter(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="broadcast_parameter", args=args, kwargs=kwargs)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        await self.engine_core.collective_rpc_async(method="update_parameter_in_bucket", args=(serialized_named_tensors, is_lora))

    async def add_lora(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="custom_add_lora", args=args, kwargs=kwargs)

    async def process_weights_after_loading(self):
        await self.engine_core.collective_rpc_async(method="process_weights_after_loading")

    async def call_engine_core_utility(self, method: str, *args):
        if hasattr(self.engine_core, "core_engines") and hasattr(self.engine_core, "_call_utility_async"):
            engines = list(getattr(self.engine_core, "core_engines", []) or [])
            if len(engines) > 1:
                return await asyncio.gather(*[
                    self.engine_core._call_utility_async(method, *args, engine=engine)
                    for engine in engines
                ])
        if hasattr(self.engine_core, "call_utility_async"):
            return await self.engine_core.call_utility_async(method, *args)
        raise AttributeError(f"Engine core client does not support utility method calls: {type(self.engine_core)}")

    async def _refresh_stats_logger_snapshots(self) -> None:
        logger_manager = getattr(self, "logger_manager", None)
        if logger_manager is None or not hasattr(self, "call_engine_core_utility"):
            return

        try:
            raw_snapshot = await self.call_engine_core_utility("roll_get_kv_cache_snapshot")
        except Exception:
            logger.debug("Failed to refresh engine-core KV snapshot for periodic stats logging.", exc_info=True)
            return

        per_engine_logger_dict = getattr(logger_manager, "per_engine_logger_dict", {})
        for per_engine_loggers in per_engine_logger_dict.values():
            for stat_logger in per_engine_loggers:
                setattr(stat_logger, "_roll_engine_core_snapshot", None)

        for engine_index, snapshot in enumerate(self._normalize_engine_snapshots(raw_snapshot)):
            if not snapshot.get("available", False):
                continue
            normalized_snapshot = dict(snapshot)
            normalized_snapshot["engine"] = str(normalized_snapshot.get("engine", engine_index))
            for stat_logger in per_engine_logger_dict.get(int(normalized_snapshot["engine"]), []):
                setattr(stat_logger, "_roll_engine_core_snapshot", normalized_snapshot)
