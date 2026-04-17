from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind

from roll.third_party.vllm.request_timing import TracedRequestState


class _FakeLogprobsProcessor:
    prompt_logprobs = None


def test_traced_request_state_attaches_phase_timing(monkeypatch):
    monkeypatch.setattr("roll.third_party.vllm.request_timing.time.time", lambda: 110.0)

    state = TracedRequestState(
        request_id="req-1",
        parent_req=None,
        request_index=0,
        lora_name=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
        prompt="hello",
        prompt_token_ids=[1, 2, 3],
        logprobs_processor=_FakeLogprobsProcessor(),
        detokenizer=None,
        max_tokens_param=16,
        arrival_time=100.0,
        queue=None,
        log_stats=True,
    )
    assert state.stats is not None
    state.stats.queued_ts = 1.0
    state.stats.scheduled_ts = 2.5
    state.stats.first_token_ts = 4.0
    state.stats.last_token_ts = 7.0
    state.stats.first_token_latency = 4.5
    state.stats.num_generation_tokens = 5
    state.num_cached_tokens = 2

    request_output = state._new_request_output(
        request_id="req-1",
        outputs=[
            CompletionOutput(
                index=0,
                text="world",
                token_ids=[10, 11, 12],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        finished=True,
        kv_transfer_params=None,
    )

    assert isinstance(request_output, RequestOutput)
    phase_timing = getattr(request_output, "roll_phase_timing", None)
    assert phase_timing is not None
    assert phase_timing["source"] == "vllm_v1_request_state"
    assert phase_timing["unit"] == "seconds"
    assert phase_timing["queue_time"] == 1.5
    assert phase_timing["prefill_time"] == 1.5
    assert phase_timing["decode_time"] == 3.0
    assert phase_timing["inference_time"] == 4.5
    assert phase_timing["e2e_time"] == 10.0
    assert phase_timing["time_to_first_token"] == 4.5
    assert phase_timing["prompt_tokens"] == 3
    assert phase_timing["output_tokens"] == 5
    assert phase_timing["cached_tokens"] == 2
