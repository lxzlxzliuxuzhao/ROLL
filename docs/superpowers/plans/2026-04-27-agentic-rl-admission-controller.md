# Implementation Plan: Agentic RL Admission Controller

**Spec**: [2026-04-27-agentic-rl-admission-controller-design.md](../specs/2026-04-27-agentic-rl-admission-controller-design.md)  
**Created**: 2026-04-27  
**Status**: Ready for implementation

## Overview

This plan implements the Agentic RL Admission Controller as specified in the design doc. The controller dynamically adjusts `active_request_cap` based on dual-beacon signals (throughput + KV memory) to maximize decode throughput while preventing thrashing.

**Architecture**: Option A (primary) + Option D (emergency fallback)
- Primary: AdmissionController as submodule in RolloutScheduler
- Fallback: Emergency KV-triggered intervention via AgenticKVRuntime

**Implementation Strategy**: Phase 1 (threshold rules + shadow PID) → Phase 2 (real PID)

## Dependencies

- LMCache integration (already in codebase)
- Tracing infrastructure (roll/utils/tracing/)
- AgenticKVRuntime (roll/pipeline/agentic/akv/runtime.py)
- RolloutScheduler (roll/pipeline/agentic/scheduler/)

## Task Breakdown

### Phase 0: Foundation (Tasks 1-3)

#### Task 1: Create admission controller module structure
**Files to create**:
- `roll/pipeline/agentic/admission/__init__.py`
- `roll/pipeline/agentic/admission/config.py`
- `roll/pipeline/agentic/admission/snapshots.py`

**What to do**:
1. Create the `admission/` directory under `roll/pipeline/agentic/`
2. Define `AdmissionControllerConfig` dataclass in `config.py` with all parameters from §2.3 of spec
3. Define data structures in `snapshots.py`:
   - `BeaconSnapshot` (timestamp, decode_throughput_tps, throughput_delta, kv_free_ratio, inflight_requests, waiting_sessions, ready_to_resume_sessions)
   - `PoolStateSnapshot` (session_states: List[SessionStateRecord])
   - `SessionStateRecord` (id, state, tool_name, wait_started_at)
   - `AdmissionDecision` (target_active_cap, swap_out_pressure, reason, shadow_pid_suggestion)

**Acceptance**:
- All dataclasses have proper type hints
- Config has sensible defaults matching spec §2.3
- Can import from `roll.pipeline.agentic.admission`

**Estimated effort**: 30 minutes

---

#### Task 2: Extend AgenticKVRuntime with query methods
**Files to modify**:
- `roll/pipeline/agentic/akv/runtime.py`

**What to do**:
1. Add `query_session_states() -> List[SessionStateRecord]` method
   - Iterate over all sessions in the runtime
   - For each session, extract: id, state (RUNNING/TOOL_WAIT/READY_TO_RESUME), tool_name (if TOOL_WAIT), wait_started_at
   - Return as list of SessionStateRecord
2. Add `get_kv_free_ratio() -> float` method
   - Query LMCache for current HBM usage
   - Return `free_bytes / total_bytes`
3. Add `request_swap_out(session_ids: List[str], priority: str) -> None` method
   - Mark sessions for eviction with given priority
   - Integrate with existing LMCache eviction logic

**Acceptance**:
- Methods return correct data types
- `query_session_states()` accurately reflects current session states
- `get_kv_free_ratio()` returns value in [0.0, 1.0]
- `request_swap_out()` triggers LMCache eviction

**Estimated effort**: 1 hour

---

#### Task 3: Implement BeaconCollector
**Files to create**:
- `roll/pipeline/agentic/admission/beacon_collector.py`

**What to do**:
1. Create `BeaconCollector` class with:
   - `__init__(window_seconds: float, akv_runtime: AgenticKVRuntime)`
   - `_throughput_history: deque` (stores (timestamp, token_count) tuples)
   - `_last_throughput: Optional[float]`
2. Implement `collect() -> BeaconSnapshot`:
   - Query `akv_runtime.query_session_states()` for session counts
   - Query `akv_runtime.get_kv_free_ratio()` for KV memory
   - Calculate decode throughput from recent token counts (sliding window average)
   - Calculate throughput_delta as `(current - last) / last` if last exists
   - Return BeaconSnapshot
3. Implement `record_tokens(count: int)` to update throughput history

**Acceptance**:
- Throughput calculation uses only decode tokens (not prefill)
- Sliding window correctly drops old samples beyond `window_seconds`
- throughput_delta is None on first call, then valid float

**Estimated effort**: 45 minutes

---

### Phase 1: Core Control Logic (Tasks 4-7)

#### Task 4: Implement StateClassifier
**Files to create**:
- `roll/pipeline/agentic/admission/state_classifier.py`

**What to do**:
1. Define `SystemState` enum: COLD_START, PARETO, THRASHING, TOOL_WAIT_BUBBLE
2. Implement pure function `classify_state(beacon: BeaconSnapshot, config: AdmissionControllerConfig) -> SystemState`:
   - Use thresholds from config (e.g., `pareto_throughput_min`, `thrashing_kv_threshold`)
   - Apply decision tree from spec §3.2:
     - COLD_START: low throughput + positive delta + high KV
     - PARETO: high throughput + stable delta + medium/low KV
     - THRASHING: low throughput + negative delta + low KV
     - TOOL_WAIT_BUBBLE: low throughput + negative/zero delta + high KV

**Acceptance**:
- Function is pure (no side effects)
- All four states can be triggered by appropriate beacon values
- Edge cases handled (e.g., throughput_delta=None)

**Estimated effort**: 30 minutes

---

#### Task 5: Implement ToolDurationEstimator
**Files to create**:
- `roll/pipeline/agentic/admission/tool_duration_estimator.py`

**What to do**:
1. Create `ToolDurationEstimator` class with:
   - `_history: Dict[str, float]` (tool_name -> EWMA duration)
   - `_alpha: float = 0.3` (EWMA smoothing factor)
2. Implement `record_completion(tool_name: str, duration_seconds: float)`:
   - Update EWMA: `history[tool] = alpha * duration + (1-alpha) * history.get(tool, duration)`
3. Implement `estimate_remaining(tool_name: str, elapsed: float, fallback: float) -> float`:
   - If tool_name in history: return `max(0, history[tool] - elapsed)`
   - Else: return `max(0, fallback - elapsed)`

**Acceptance**:
- EWMA correctly smooths durations over multiple calls
- Cold start (no history) uses fallback value
- Negative remaining durations clamped to 0

**Estimated effort**: 30 minutes

---

#### Task 6: Implement SessionPriority scorer
**Files to create**:
- `roll/pipeline/agentic/admission/session_priority.py`

**What to do**:
1. Define `SessionPriority` enum: P0_RUNNING, P1_READY, P2_SHORT_WAIT, P3_LONG_WAIT
2. Implement `assign_priority(session: SessionStateRecord, estimator: ToolDurationEstimator, config: AdmissionControllerConfig) -> SessionPriority`:
   - P0: state == RUNNING
   - P1: state == READY_TO_RESUME
   - P2/P3: state == TOOL_WAIT, compute remaining_ratio:
     - `elapsed = now - wait_started_at`
     - `expected_remaining = estimator.estimate_remaining(tool_name, elapsed, config.fallback_wait_seconds)`
     - `expected_total = elapsed + expected_remaining`
     - `remaining_ratio = expected_remaining / expected_total`
     - P2 if `remaining_ratio < SHORT_RATIO (0.3)`, else P3
3. Implement `score_sessions(sessions: List[SessionStateRecord], ...) -> Dict[str, SessionPriority]`

**Acceptance**:
- P0/P1 assigned based on state machine facts
- P2/P3 use relative remaining_ratio, not absolute time
- Cold start (no tool history) defaults to P3

**Estimated effort**: 45 minutes

---

#### Task 7: Implement BudgetAllocator (Phase 1: threshold rules)
**Files to create**:
- `roll/pipeline/agentic/admission/budget_allocator.py`

**What to do**:
1. Create `BudgetAllocator` class with threshold-based logic
2. Implement `decide(beacon: BeaconSnapshot, state: SystemState, current_cap: int, config: AdmissionControllerConfig) -> AdmissionDecision`:
   - COLD_START: increase cap by step (up to max)
   - PARETO: hold cap steady
   - THRASHING: decrease cap by step (down to min)
   - TOOL_WAIT_BUBBLE: decrease cap + set swap_out_pressure=HIGH
3. Clamp target_cap to [min, max] from config
4. Return AdmissionDecision with reason string

**Acceptance**:
- Cap changes are gradual (step-based, not jumps)
- TOOL_WAIT_BUBBLE triggers swap_out_pressure
- Decision includes human-readable reason

**Estimated effort**: 45 minutes

---

### Phase 2: Integration & Emergency Path (Tasks 8-10)

#### Task 8: Implement ShadowPID controller
**Files to create**:
- `roll/pipeline/agentic/admission/shadow_pid.py`

**What to do**:
1. Create `ShadowPDController` class (PD only, no integral term):
   - `__init__(kp: float, kd: float, target_throughput: float)`
   - `_last_error: Optional[float]`
   - `_last_time: Optional[float]`
2. Implement `compute(current_throughput: float, dt: float) -> float`:
   - `error = target_throughput - current_throughput`
   - `p_term = kp * error`
   - `d_term = kd * (error - last_error) / dt` if last_error exists
   - Return `p_term + d_term` (suggested cap adjustment)
3. Implement `update_target(new_target: float)` for adaptive throughput

**Acceptance**:
- PD formula matches control theory conventions
- First call (no history) only uses P term
- Output is logged but NOT applied in Phase 1

**Estimated effort**: 30 minutes

---

#### Task 9: Implement AdmissionController main orchestrator
**Files to create**:
- `roll/pipeline/agentic/admission/controller.py`

**What to do**:
1. Create `AdmissionController` class integrating all components:
   - `__init__(config: AdmissionControllerConfig, akv_runtime: AgenticKVRuntime, tracer: Tracer)`
   - Instantiate: BeaconCollector, StateClassifier, BudgetAllocator, ToolDurationEstimator, SessionPriority, ShadowPID
   - `_current_cap: int = config.active_request_cap_init`
   - `_emergency_mode: bool = False`
2. Implement `step() -> int`:
   - Collect beacon
   - Check emergency conditions (KV < low_watermark)
   - If emergency: set cap=min, request swap_out P3+P2, set emergency_mode=True
   - If not emergency and was in emergency: check KV > high_watermark to exit
   - If normal: classify state, decide budget, update cap
   - If shadow_pid enabled: compute PD suggestion and log to tracing
   - Record decision to tracing span
   - Return new cap
3. Implement `record_tool_completion(tool_name: str, duration: float)` to update estimator

**Acceptance**:
- Emergency path overrides window-based control
- Hysteresis prevents emergency mode flapping (low/high watermarks)
- Shadow PID runs in parallel but doesn't affect cap
- All decisions traced with structured data

**Estimated effort**: 1.5 hours

---

#### Task 10: Integrate AdmissionController into RolloutScheduler
**Files to modify**:
- `roll/pipeline/agentic/scheduler/rollout_scheduler.py` (or equivalent)
- `roll/pipeline/agentic/agentic_config.py`

**What to do**:
1. Add `admission_controller_config: Optional[AdmissionControllerConfig]` to AgenticConfig
2. In RolloutScheduler.__init__:
   - If config.admission_controller_config and config.admission_controller_config.enable:
     - Instantiate AdmissionController
   - Else: set to None (disabled)
3. In RolloutScheduler main loop (where admission decisions happen):
   - If controller is not None:
     - Call `new_cap = controller.step()`
     - Update `self._active_request_cap = new_cap`
   - Else: use static cap from config
4. Hook tool completion events to `controller.record_tool_completion()`

**Acceptance**:
- Controller is optional (enable=False disables entirely)
- Scheduler respects dynamic cap from controller
- Tool completions feed back to estimator
- No behavior change when disabled

**Estimated effort**: 1 hour

---

### Phase 3: LMCache Integration (Tasks 11-12)

#### Task 11: Implement proactive swap-out in AdmissionController
**Files to modify**:
- `roll/pipeline/agentic/admission/controller.py`

**What to do**:
1. In `step()` method, after budget decision:
   - If `decision.swap_out_pressure == HIGH`:
     - Query `akv_runtime.query_session_states()`
     - Score sessions with SessionPriority
     - Identify P3 sessions (long wait candidates)
     - Call `akv_runtime.request_swap_out(p3_session_ids, priority="LOW")`
2. Add tracing for swap-out actions

**Acceptance**:
- Swap-out only triggered when decision says so
- Only P3 sessions swapped proactively
- P2 sessions only swapped in emergency mode
- Swap-out events appear in tracing

**Estimated effort**: 45 minutes

---

#### Task 12: Verify LMCache tiered residency
**Files to check**:
- `roll/pipeline/agentic/akv/runtime.py` (LMCache integration)

**What to do**:
1. Verify that LMCache already supports priority-based eviction
2. Ensure P0 (RUNNING) sessions are never evicted
3. Ensure P1 (READY_TO_RESUME) sessions are evicted only under pressure
4. Ensure P2/P3 sessions can be evicted on request
5. If priority system doesn't exist, add session priority metadata to LMCache calls

**Acceptance**:
- LMCache respects session priorities
- P0 sessions stay in HBM under all conditions
- Eviction order: P3 → P2 → P1 (never P0)

**Estimated effort**: 1 hour (may require LMCache API changes)

---

### Phase 4: Adaptive Throughput Target (Tasks 13-14)

#### Task 13: Implement AdaptiveThroughputTarget
**Files to create**:
- `roll/pipeline/agentic/admission/adaptive_target.py`

**What to do**:
1. Create `AdaptiveThroughputTarget` class with TCP-inspired AIMD:
   - `__init__(initial_target: float, increase_step: float, decrease_factor: float)`
   - `_current_target: float`
   - `_last_state: Optional[SystemState]`
2. Implement `update(state: SystemState) -> float`:
   - If state == PARETO: `target += increase_step` (additive increase)
   - If state == THRASHING: `target *= decrease_factor` (multiplicative decrease)
   - Else: hold target
   - Return new target
3. Implement `get_target() -> float`

**Acceptance**:
- Target increases slowly in PARETO (additive)
- Target decreases quickly in THRASHING (multiplicative)
- Target stable in other states

**Estimated effort**: 30 minutes

---

#### Task 14: Integrate adaptive target into controller
**Files to modify**:
- `roll/pipeline/agentic/admission/controller.py`
- `roll/pipeline/agentic/admission/config.py`

**What to do**:
1. Add adaptive target config to AdmissionControllerConfig:
   - `adaptive_target_enable: bool = True`
   - `adaptive_target_initial: float = 1000.0` (tokens/sec)
   - `adaptive_target_increase_step: float = 50.0`
   - `adaptive_target_decrease_factor: float = 0.8`
2. In AdmissionController.__init__:
   - Instantiate AdaptiveThroughputTarget if enabled
3. In step() method:
   - After state classification, call `target.update(state)`
   - Update ShadowPID target: `shadow_pid.update_target(target.get_target())`
4. Add warmup logic (optional):
   - On first N steps, run cap sweep to find initial target
   - Set adaptive target to observed peak throughput

**Acceptance**:
- Adaptive target adjusts based on system state
- Shadow PID uses adaptive target (not static)
- Warmup (if implemented) finds reasonable initial target

**Estimated effort**: 1 hour

---

### Phase 5: Testing & Validation (Tasks 15-17)

#### Task 15: Write unit tests for core components
**Files to create**:
- `tests/pipeline/agentic/admission/test_beacon_collector.py`
- `tests/pipeline/agentic/admission/test_state_classifier.py`
- `tests/pipeline/agentic/admission/test_budget_allocator.py`
- `tests/pipeline/agentic/admission/test_tool_duration_estimator.py`
- `tests/pipeline/agentic/admission/test_session_priority.py`
- `tests/pipeline/agentic/admission/test_shadow_pid.py`

**What to do**:
1. For each component, write tests covering:
   - Happy path (normal inputs)
   - Edge cases (empty history, None values, boundary conditions)
   - State transitions (e.g., COLD_START → PARETO → THRASHING)
2. Use pytest fixtures for common test data (BeaconSnapshot, config)
3. Mock AgenticKVRuntime for BeaconCollector tests

**Acceptance**:
- All tests pass
- Coverage > 80% for admission/ module
- Tests are fast (< 1s total)

**Estimated effort**: 2 hours

---

#### Task 16: Write integration test for AdmissionController
**Files to create**:
- `tests/pipeline/agentic/admission/test_controller_integration.py`

**What to do**:
1. Create mock AgenticKVRuntime with controllable state
2. Simulate scenarios:
   - Cold start → ramp up cap
   - PARETO → hold cap
   - THRASHING → reduce cap
   - Emergency KV low → drop to min + swap out
   - Tool wait bubble → reduce cap + swap out
3. Verify cap changes, swap-out calls, tracing output
4. Verify shadow PID runs but doesn't affect cap

**Acceptance**:
- All scenarios produce expected cap changes
- Emergency mode triggers correctly
- Shadow PID logs appear in tracing
- No crashes or exceptions

**Estimated effort**: 1.5 hours

---

#### Task 17: End-to-end test with RolloutScheduler
**Files to create**:
- `tests/pipeline/agentic/test_scheduler_with_admission.py`

**What to do**:
1. Create minimal RolloutScheduler setup with real AgenticKVRuntime
2. Enable AdmissionController in config
3. Run a small rollout (e.g., 10 sessions, simple tool calls)
4. Verify:
   - Controller step() is called periodically
   - Cap changes affect actual admission decisions
   - Tool completions update estimator
   - Tracing spans contain admission decisions
5. Compare throughput with/without controller (sanity check)

**Acceptance**:
- Test runs without errors
- Controller integrates cleanly with scheduler
- Tracing output is complete and parsable
- Throughput is reasonable (not degraded)

**Estimated effort**: 2 hours

---

### Phase 6: Documentation & Observability (Tasks 18-19)

#### Task 18: Add tracing instrumentation
**Files to modify**:
- `roll/pipeline/agentic/admission/controller.py`
- `roll/pipeline/agentic/admission/beacon_collector.py`

**What to do**:
1. Wrap `AdmissionController.step()` in tracing span:
   - Span name: "admission_controller.step"
   - Attributes: beacon values, state, decision, cap_before, cap_after
2. Wrap `BeaconCollector.collect()` in span:
   - Span name: "admission_controller.collect_beacons"
   - Attributes: throughput, kv_free_ratio, session_counts
3. Add shadow PID output to span attributes
4. Add swap-out actions to span attributes

**Acceptance**:
- All admission decisions appear in tracing
- Spans are nested correctly (collect → classify → decide)
- Shadow PID suggestions are logged
- Tracing overhead is minimal (< 1ms per step)

**Estimated effort**: 1 hour

---

#### Task 19: Write user-facing documentation
**Files to create**:
- `docs/agentic_pipeline/admission_controller.md`

**What to do**:
1. Write guide covering:
   - What the admission controller does (high-level)
   - When to enable it (use cases)
   - How to configure it (parameter guide)
   - How to interpret tracing output
   - Troubleshooting common issues
2. Include example config snippets
3. Link to design spec for details

**Acceptance**:
- Documentation is clear and actionable
- Examples are copy-pasteable
- Covers both Phase 1 (threshold rules) and Phase 2 (PID)

**Estimated effort**: 1.5 hours

---

## Execution Strategy

**Recommended approach**: Inline execution with checkpoints

1. **Checkpoint 1** (Tasks 1-3): Foundation
   - Create module structure, extend AKV runtime, implement beacon collector
   - Verify: Can collect beacons from running system

2. **Checkpoint 2** (Tasks 4-7): Core logic
   - Implement state classifier, tool estimator, session priority, budget allocator
   - Verify: Unit tests pass, logic is sound

3. **Checkpoint 3** (Tasks 8-10): Integration
   - Implement shadow PID, main controller, integrate into scheduler
   - Verify: Controller runs in scheduler, cap changes dynamically

4. **Checkpoint 4** (Tasks 11-12): LMCache
   - Add proactive swap-out, verify tiered residency
   - Verify: Sessions are swapped correctly

5. **Checkpoint 5** (Tasks 13-14): Adaptive target
   - Implement AIMD target, integrate into controller
   - Verify: Target adapts to system state

6. **Checkpoint 6** (Tasks 15-17): Testing
   - Write unit tests, integration tests, e2e tests
   - Verify: All tests pass, coverage is good

7. **Checkpoint 7** (Tasks 18-19): Observability
   - Add tracing, write documentation
   - Verify: Tracing is complete, docs are clear

**Estimated total effort**: 18-20 hours

## Risks & Mitigations

1. **Risk**: LMCache API doesn't support priority-based eviction
   - **Mitigation**: Task 12 checks this early; if needed, extend LMCache API

2. **Risk**: Throughput calculation is noisy or inaccurate
   - **Mitigation**: Use sliding window average (Task 3); tune window_seconds

3. **Risk**: Shadow PID suggests unstable cap changes
   - **Mitigation**: Phase 1 uses threshold rules; PID is shadow-only until validated

4. **Risk**: Emergency mode triggers too frequently
   - **Mitigation**: Tune watermarks (low=0.10, high=0.25); add hysteresis

5. **Risk**: Tool duration estimates are poor for cold start
   - **Mitigation**: Use conservative fallback (10s); collect data over time

## Success Criteria

- [ ] Controller can be enabled/disabled via config flag
- [ ] Cap adjusts dynamically based on system state
- [ ] Emergency mode prevents OOM in low-KV scenarios
- [ ] Proactive swap-out reduces thrashing in tool-wait-heavy workloads
- [ ] Shadow PID logs provide data for Phase 2 tuning
- [ ] All tests pass (unit + integration + e2e)
- [ ] Tracing output is complete and actionable
- [ ] Documentation enables users to configure and debug controller

## Phase 2 Evolution Path

After Phase 1 is stable and shadow PID data is collected:

1. Analyze shadow PID logs to tune kp, kd parameters
2. Compare shadow PID suggestions vs threshold rule decisions
3. If shadow PID is more stable: switch to real PID in BudgetAllocator
4. Add integral term (PID) if steady-state error is observed
5. Deprecate threshold rules, keep only as fallback

**Estimated Phase 2 effort**: 4-6 hours (analysis + tuning + switch)
