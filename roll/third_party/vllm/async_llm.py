import asyncio

from vllm.v1.engine.async_llm import AsyncLLM

from roll.third_party.vllm.request_timing import TracedOutputProcessor


class CustomAsyncLLM(AsyncLLM):
    def _ensure_traced_output_processor(self):
        if isinstance(self.output_processor, TracedOutputProcessor):
            return

        previous_output_processor = self.output_processor
        traced_output_processor = TracedOutputProcessor(self.tokenizer, log_stats=self.log_stats)
        traced_output_processor.tracer = previous_output_processor.tracer
        self.output_processor = traced_output_processor

    def _run_output_handler(self):
        self._ensure_traced_output_processor()
        return super()._run_output_handler()

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
