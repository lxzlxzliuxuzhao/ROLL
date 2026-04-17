import time
from typing import Optional

from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector, RequestState
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine import EngineCoreRequest


class TracedRequestState(RequestState):
    @staticmethod
    def _interval_seconds(start: Optional[float], end: Optional[float]) -> Optional[float]:
        if start is None or end is None:
            return None
        return max(float(end - start), 0.0)

    def _build_phase_timing(self, *, finished: bool) -> Optional[dict]:
        if not finished or self.stats is None:
            return None

        queued_ts = self.stats.queued_ts or None
        scheduled_ts = self.stats.scheduled_ts or None
        first_token_ts = self.stats.first_token_ts or None
        last_token_ts = self.stats.last_token_ts or None
        finished_time = time.time()

        return {
            "source": "vllm_v1_request_state",
            "unit": "seconds",
            "queue_time": self._interval_seconds(queued_ts, scheduled_ts),
            "prefill_time": self._interval_seconds(scheduled_ts, first_token_ts),
            "decode_time": self._interval_seconds(first_token_ts, last_token_ts),
            "inference_time": self._interval_seconds(scheduled_ts, last_token_ts),
            "e2e_time": self._interval_seconds(self.stats.arrival_time, finished_time),
            "time_to_first_token": (
                max(float(self.stats.first_token_latency), 0.0)
                if self.stats.first_token_latency
                else None
            ),
            "prompt_tokens": len(self.prompt_token_ids),
            "output_tokens": self.stats.num_generation_tokens,
            "cached_tokens": self.num_cached_tokens,
        }

    def _new_request_output(
        self,
        request_id: str,
        outputs,
        finished: bool,
        kv_transfer_params: Optional[dict] = None,
    ) -> RequestOutput | PoolingRequestOutput:
        request_output = super()._new_request_output(
            request_id=request_id,
            outputs=outputs,
            finished=finished,
            kv_transfer_params=kv_transfer_params,
        )

        if isinstance(request_output, RequestOutput):
            phase_timing = self._build_phase_timing(finished=finished)
            if phase_timing is not None:
                setattr(request_output, "roll_phase_timing", phase_timing)

        return request_output


class TracedOutputProcessor(OutputProcessor):
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: Optional[str],
        parent_req: Optional[ParentRequest] = None,
        request_index: int = 0,
        queue: Optional[RequestOutputCollector] = None,
    ) -> None:
        request_id = request.request_id
        if request_id in self.request_states:
            raise ValueError(f"Request id {request_id} already running.")

        tokenizer = None if not self.tokenizer else self.tokenizer.get_lora_tokenizer(request.lora_request)

        req_state = TracedRequestState.from_new_request(
            tokenizer=tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
        )
        self.request_states[request_id] = req_state
        self.lora_states.add_request(req_state)
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req
