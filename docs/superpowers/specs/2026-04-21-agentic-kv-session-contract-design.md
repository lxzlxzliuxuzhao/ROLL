# Agentic RL KV Session Contract Design

## Status

Approved for spec drafting on 2026-04-21.

## Goal

Define an Agentic-RL-native KV cache management contract for multi-turn tool-using trajectories. This contract must describe:

- what a trajectory-scoped KV state means semantically
- when that state leaves and re-enters the inference backend active set
- what unload, restore, and resume mean at the semantic layer
- what correctness guarantees must hold across tool wait, env wait, and retry/replay

The contract must be backend-agnostic at the semantic layer, while still allowing a concrete `vLLM` adapter contract. `LMCache` is treated as one possible implementation path, not as the semantic model.

## Non-Goals

This spec does not attempt to:

- make cross-request KV reuse a first-class managed object
- define a physical memory layout in HBM, CPU RAM, or external cache
- require any specific cache backend such as `LMCache`
- support branching trajectory futures in v1
- define low-level transport details such as PCIe copy scheduling or buffer formats

Natural shared-prefix hits such as `system prompt` reuse are allowed, but they are not actively managed by the semantic layer in v1.

## Design Choice

This design uses:

- a backend-agnostic semantic contract centered on `Trajectory KV Session`
- a separate adapter contract for `vLLM`

This is intentionally not a cache-backend-first design. Agentic RL semantics come first; backend mechanisms implement them.

## Core Model

### `Trajectory KV Session`

`Trajectory KV Session` is the primary semantic object.

It represents the recoverable generation state of one trajectory lineage under one model snapshot. It is not defined as:

- a specific block range in HBM
- one object stored inside `LMCache`
- a single `vLLM request_id`

Instead, it is a unified recoverable semantic object. If the system claims that a session can continue, it must be able to continue from a legal boundary with `token-exact` behavior, regardless of whether the implementation path is direct KV reuse, offloaded KV restore, or exact recomputation.

### `Resume Point`

`Resume Point` is not a separate KV asset. It is a semantic boundary marker attached to a `Trajectory KV Session`.

Each `Resume Point` identifies a safe continuation boundary where the session may later be restored and resumed. In v1, the primary continuation boundary is request-scoped: one inference request ends, but the trajectory has not ended. This is the key Agentic RL case for tool and env waits.

It refers to:

- one `Trajectory KV Session`
- one legal lineage position
- one concrete turn or chunk boundary
- one recoverable execution state boundary

It does not own physical KV state. The underlying implementation may support a `Resume Point` by:

- keeping relevant private KV materialized in HBM
- externalizing recoverable state to CPU or external cache
- dropping physical KV and later reconstructing exact state by recomputation

### Boundary Generation

`Resume Point`s are created automatically at:

- every `turn` boundary
- every request-end boundary that `env manager` declares as a legal continuation boundary for the same trajectory session

In the `vLLM` setting, the most important v1 case is:

- the request terminates because a tool-call terminator or equivalent EOS condition is emitted
- the current inference request ends
- the `Trajectory KV Session` does not end
- after tool or env output is appended back into the prompt, a new inference request continues the same trajectory session

So the semantic contract is not built around arbitrary backend chunking. It is built around legal request-end continuation boundaries inside one still-live trajectory.

`env manager` remains the authority on which request-end boundaries are legal continuation points for the trajectory. Output parsers or backend heuristics may observe candidate boundaries, but they do not define them in the semantic contract.

### Stable Identity vs Execution Identity

`request_id` is not a semantic primary key. It is an adapter-local execution identifier.

One `Trajectory KV Session` may correspond to multiple backend requests across:

- multiple turns
- retry/replay
- restore and resume cycles

Therefore, the semantic contract requires a stable session identity above backend request identity.

## Ownership and Lifecycle

### Ownership

The semantic layer must know which KV state belongs to which `Trajectory KV Session`.

This ownership is trajectory-scoped first. The v1 contract does not require first-class management of cross-request shared cache objects. If static prefixes happen to stay resident and naturally hit later, that is allowed, but it is not the main object being scheduled.

### Lifecycle States

The semantic contract is lifecycle-driven. v1 uses four session states:

- `running`
- `waiting_external`
- `finished`
- `invalidated`

#### `running`

`running` means the session currently belongs to the inference backend active request set.

This definition is intentionally aligned with backend scheduling semantics such as `vLLM active set`. It does not mean:

- the session is physically in HBM
- the session is physically in CPU RAM
- the session has or has not been offloaded

While a session is `running`, it is in use and is not eligible for `unload`.

#### `waiting_external`

`waiting_external` means:

- the current inference request has ended
- the trajectory has not ended
- the session is waiting for tool or env output before continuation

This is the primary Agentic RL scheduling state in v1.

Each transition into `waiting_external` automatically creates a new `Resume Point` at the request-end continuation boundary.

Only `waiting_external` sessions participate in watermark-driven `unload`.

#### `finished`

`finished` means the trajectory has completed normally and the session no longer has a continuation obligation.

Owned resources may be released immediately when the session enters `finished`.

#### `invalidated`

`invalidated` means the system can no longer legally continue the session.

This is not a normal scheduling state. It is a terminal state caused by incompatibility or loss of proof for `token-exact` continuation.

### Lifecycle Transitions

The main v1 lifecycle is:

- `running -> waiting_external`
- `waiting_external -> running`
- `running -> finished`
- `waiting_external -> finished`
- `running -> invalidated`
- `waiting_external -> invalidated`

Retry or replay is not a separate state. It is a `resume` transition targeting an older legal `Resume Point`.

## Events

The semantic layer is event-driven for boundary and lineage updates.

Core events:

- `resume_point_created(session, resume_point, boundary_kind)`
- `left_active(session, reason, latest_resume_point)`
- `env_result_ready(session, resume_point)`
- `retry_requested(session, target_resume_point)`
- `session_completed(session)`
- `session_invalidated(session, reason)`
- `model_snapshot_changed(snapshot)`

Important constraints:

- `left_active` must use Agentic RL reasons such as `tool_wait`, `env_wait`, or `retry_wait`
- for `tool_wait` and `env_wait`, the corresponding `left_active` event means the current inference request has ended while the trajectory session remains live
- `left_active(..., tool_wait | env_wait, ...)` implies a transition from `running` to `waiting_external`
- `env_result_ready(session, resume_point)` enables a later `waiting_external -> running` transition through `restore` and `resume`
- `session_completed(session)` implies transition to `finished` and release of owned continuation obligation
- `model_snapshot_changed` is a semantic invalidation boundary
- `retry_requested` does not create branching in v1; it initiates rollback on a single lineage

## Watermark-Based Unload Policy

v1 does not use unconditional unload on every wait. It uses a capacity-target policy over waiting sessions.

### Capacity Target

The semantic policy is defined in terms of a capacity target such as:

- minimum free blocks
- minimum free HBM percentage

The semantic layer allows a global target, but the v1 `vLLM` adapter may implement this first with per-engine or per-block-pool local targets.

### Double Watermark

The capacity target uses hysteresis:

- when free capacity drops below `low watermark`, unload of eligible waiting sessions must begin
- once unloading begins, it continues until free capacity reaches `high watermark`
- between `low watermark` and `high watermark`, the system does not oscillate between unload and no-unload behavior

### Eligible Sessions

Only `waiting_external` sessions are eligible for watermark-driven unload.

- `running` sessions cannot be unloaded
- `finished` sessions are released rather than unloaded
- `invalidated` sessions are not resumable

### Unload Requirement

If free capacity remains at or above `high watermark`, a `waiting_external` session may remain resident in HBM.

If free capacity falls below `low watermark`, the adapter must unload enough eligible `waiting_external` private KV footprint to recover free capacity toward `high watermark`.

This policy applies to private trajectory-owned KV footprint. Natural shared-prefix residency is not required to be removed.

## Command Surface

The command surface is intentionally small.

### Resource and Continuation Commands

- `unload(session, resume_point)`
- `restore(session, resume_point)`
- `resume(session, resume_point | latest)`

### Command Semantics

#### `unload(session, resume_point)`

Strong semantic guarantee:

- after success, the session's private KV footprint is no longer counted in the active HBM working set
- the targeted `Resume Point` remains semantically recoverable

For v1 tool-using trajectories, `unload` is tied to `waiting_external` and capacity pressure. When a request ends due to `tool_wait` or `env_wait`, the session becomes eligible for unload. If watermark pressure requires action, this waiting session's private KV footprint must leave the active HBM working set.

This does not prescribe a specific mechanism. It may be satisfied by:

- offload to CPU or external cache
- another backend-specific preservation strategy
- exact-state discard plus future exact recomputation, if exact continuation can still be proven later

Natural shared-prefix residency is not required to be removed by `unload`.

#### `restore(session, resume_point)`

Strong semantic guarantee:

- the adapter brings the session into a fast-resume implementation state
- this does not itself re-enter the backend active set

#### `resume(session, resume_point | latest)`

Strong semantic guarantee:

- the session re-enters execution from the specified legal boundary
- continuation must be `token-exact`

`resume` is the only command that actually continues generation. `restore` is preparatory.

## Preconditions and Postconditions

### `unload`

Preconditions:

- session is `waiting_external`
- target `Resume Point` is legal
- session is not `invalidated`

Postconditions:

- session remains recoverable
- private HBM footprint exits the active HBM working set

Additional v1 rule:

- watermark-driven unload applies only to `waiting_external`

### `restore`

Preconditions:

- session is `waiting_external`
- session is not `invalidated`
- target `Resume Point` is legal

Postconditions:

- session remains `waiting_external`
- session enters a backend-specific fast-resume state

### `resume`

Preconditions:

- session is `waiting_external`
- target `Resume Point` is legal
- adapter can prove `token-exact` continuation

Postconditions:

- session becomes `running`
- generation continues on the same legal lineage

## Correctness and Equivalence

### `token-exact`

The semantic contract requires `token-exact` continuation.

This means that resuming from a legal `Resume Point` must produce the same future token sequence as uninterrupted execution from that same boundary.

The contract therefore covers more than prefix KV. A recoverable session semantically includes all state necessary to continue exactly, including:

- model snapshot identity
- visible prefix state
- decode or sampling configuration
- RNG state
- stop-condition or parser state, if such state affects future token generation

The semantic contract does not require any specific physical restoration mechanism. It requires only exact behavioral equivalence.

### Legal Reuse Conditions

Two checks are required for direct reuse:

- `lineage legality`
- `prefix compatibility`

`lineage legality` decides whether reuse is semantically permitted.  
`prefix compatibility` decides whether direct physical reuse is possible.

Both must hold for direct reuse. If direct reuse does not hold but exact reconstruction is possible, the adapter may still satisfy `resume` by exact recomputation.

## Invalidation Rules

By default, a `Trajectory KV Session` is valid only within:

- the same rollout phase
- the same model snapshot

This default may be widened by configuration to allow reuse across rollout phases when model snapshot compatibility still holds. However, widening validity still requires compatibility of all state needed for `token-exact` continuation.

A session or its `Resume Point`s must be invalidated when:

- model snapshot compatibility is broken
- rollback to an earlier `Resume Point` succeeds and later future points are overwritten
- the adapter can no longer prove `token-exact` continuation

If exact continuation cannot be proven, the adapter must return invalidation or command failure. It must not silently perform approximate recovery.

Normal completion is not invalidation. When a session reaches `finished`, its owned continuation resources may be released because no further resume is legal or required.

## Retry and Replay

v1 supports:

- linear multi-turn continuation
- retry/replay from an earlier `Resume Point`

v1 does not support:

- branching futures from one session

If `resume` succeeds from an earlier `Resume Point`, then all later `Resume Point`s on the overwritten future become invalid. This preserves a single active lineage.

## Adapter Contract for `vLLM`

### Responsibility Split

The semantic contract defines meaning. The `vLLM` adapter maps that meaning onto backend request lifecycle.

The adapter must answer:

- which backend requests currently implement a given `Trajectory KV Session`
- when that session leaves or re-enters the `vLLM active set`
- whether continuation is satisfied by direct reuse, externalized restore, or exact recomputation
- how local capacity targets and low/high watermark policy are enforced for waiting sessions

In v1, the critical adapter case is not generic cache reuse. It is the transition:

- one `vLLM` request ends because a tool-call terminator or equivalent EOS is emitted
- the request leaves the `vLLM active set`
- the trajectory session remains live and waits for external output
- the adapter evaluates local watermark pressure for the waiting session
- if watermark pressure requires action, the adapter must ensure that the session's private KV footprint no longer occupies the active HBM working set during that wait
- later, after tool or env output is appended, a new request continues the same session from a legal `Resume Point`

### Mapping Rules

- `Trajectory KV Session` is the stable semantic object
- `vLLM request_id` is a transient execution identifier
- one session may map to many request IDs across turns and retries

The adapter therefore must maintain at least:

- `session -> request lineage`
- `request lineage -> current and historical request_id`
- `session -> latest legal Resume Point`

### `active` in `vLLM`

For the `vLLM` adapter, `active` specifically means:

- the session currently corresponds to a request in `vLLM active set`

It does not mean anything by itself about:

- HBM residency
- CPU residency
- external cache presence

This distinction is essential. Semantic activity and physical placement are different layers.

However, v1 also requires one specific coupling between the two layers: when a session is `waiting_external` and local watermark pressure requires unload, its private KV footprint must no longer remain in the active HBM working set after unload handling completes.

### `vLLM` Adapter Obligations

- preserve stable session identity above request identity
- surface transitions into and out of the active set
- enforce unload eligibility only for `waiting_external`
- enforce low/high watermark policy against local engine or block-pool capacity targets
- release owned continuation resources when session becomes `finished`
- honor resource and continuation commands
- refuse continuation when `token-exact` cannot be proven

### Backend Freedom

The adapter may satisfy commands using:

- direct KV preservation
- external cache systems such as `LMCache`
- exact recomputation

This freedom is permitted only as long as semantic guarantees remain intact.

## Capability Tiers

The semantic contract allows adapters with different strengths.

### `C0`

- supports `resume`
- may satisfy `resume` through exact recomputation

### `C1`

- supports `unload`
- can explicitly reduce private HBM footprint for `waiting_external` sessions
- enforces low/high watermark policy

### `C2`

- supports `restore`
- can prepare fast resume ahead of actual continuation

`LMCache`-based implementations are expected to target `C1` or higher, but the semantic layer does not require `LMCache`.

## Why This Contract

This contract is intentionally centered on Agentic RL semantics:

- trajectory-scoped ownership
- leave and re-enter active set around tool and env waits
- `running / waiting_external / finished / invalidated` lifecycle
- watermark-driven unload of waiting sessions
- explicit unload and restore primitives
- exact resume correctness
- linear retry/replay without branching

It gives upper layers a native language for KV cache scheduling in Agentic RL, while allowing lower layers to implement that language with different cache and recomputation strategies.
