"""
Naming constants for tracing spans and metrics.

This module defines standardized naming conventions for:
- Span names (trace-level, hierarchical with dot notation)
- Metrics keys (aggregated statistics level)
- Trajectory-level metrics keys (per-trajectory statistics)

Naming conventions:
- Span names: `<domain>.<operation>` (e.g., `weight_sync.sync_weights`)
- Timing metrics: `timing.<stage>` (e.g., `timing.weight_sync`)
- Throughput metrics: `throughput.<metric>` (e.g., `throughput.tokens_per_second`)
- Batch metrics: `batch.<metric>` (e.g., `batch.add_count`)
- Trajectory metrics: `trajectory.<metric>` (e.g., `trajectory.total_time`)
"""

# ==============================================================================
# Batch-level Span Names (driver process)
# ==============================================================================

# Pipeline
SPAN_PIPELINE_STEP = "pipeline.step"

# Weight Synchronization
SPAN_WEIGHT_SYNC_OFFLOAD_TRAIN = "weight_sync.offload_train"
SPAN_WEIGHT_SYNC_SYNC_WEIGHTS = "weight_sync.sync_weights"
SPAN_WEIGHT_SYNC_OFFLOAD_INFER = "weight_sync.offload_infer"
SPAN_WEIGHT_SYNC_LOAD_INFER = "weight_sync.load_infer"

# Scheduler
SPAN_SCHEDULER_SUSPEND = "scheduler.suspend"
SPAN_SCHEDULER_SUSPEND_FOR_TRAIN = "scheduler.suspend_for_train"
SPAN_SCHEDULER_SHRINK = "scheduler.shrink"
SPAN_SCHEDULER_EXPAND = "scheduler.expand"

# Rollout
SPAN_ROLLOUT_GET_BATCH = "rollout.get_batch"
SPAN_ROLLOUT_PUT_BATCH = "rollout.put_batch"
SPAN_ROLLOUT_WAIT_WORKER = "rollout.wait_worker"
SPAN_ROLLOUT_COLLECT_BATCH = "rollout.collect_batch"

# Reward
SPAN_REWARD_REF_LOGPROB = "reward.ref_logprob"
SPAN_REWARD_COMPUTE = "reward.compute"
SPAN_REWARD_TOKEN_LEVEL = "reward.token_level"

# Policy Evaluation
SPAN_POLICY_EVAL_LOGPROB = "policy_eval.logprob"

# Advantage
SPAN_ADVANTAGE_COMPUTE = "advantage.compute"

# Training
SPAN_TRAINING_ACTOR_UPDATE = "training.actor_update"


# ==============================================================================
# Trajectory-level Span Names (EnvironmentWorker process)
# ==============================================================================

# Trajectory lifecycle
SPAN_TRAJECTORY_LIFETIME = "trajectory.lifetime"
SPAN_TRAJECTORY_STEP = "trajectory.step"

# Inference
SPAN_INFERENCE_GENERATE = "inference.generate"
SPAN_INFERENCE_REQUEST = "inference.request"
SPAN_INFERENCE_PREFILL = "inference.prefill"
SPAN_INFERENCE_DECODE = "inference.decode"
SPAN_INFERENCE_OVERHEAD = "inference.overhead"
SPAN_INFERENCE_METRICS = "inference.metrics"

# Tool calls
SPAN_TRAJECTORY_TOOL_CALL = "trajectory.tool_call"

# Cache
SPAN_CACHE_EVICTION = "cache.eviction"
SPAN_CACHE_PREFETCH = "cache.prefetch"

# Environment
SPAN_ENV_RESET = "env.reset"
SPAN_ENV_STEP = "env.step"
SPAN_ENV_FORMAT_RESPONSE = "env.format_response"
SPAN_ENV_FETCH_REQUEST = "env.fetch_request"
SPAN_ENV_CHECK_TERMINATION = "env.check_termination"
SPAN_ENV_PARSE_REQUEST = "env.parse_request"
SPAN_ENV_REWARD_TEST = "env.reward_test"
SPAN_ENV_RESTART_SESSION = "env.restart_session"
SPAN_ENV_CLOSE = "env.close"
SPAN_ENV_START_AGENT = "env.start_agent"
SPAN_ENV_FETCH_INIT_REQUEST = "env.fetch_init_request"
SPAN_ENV_PARSE_INIT_REQUEST = "env.parse_init_request"


# ==============================================================================
# Phase Definitions (for stage grouping)
# ==============================================================================

PHASE_PIPELINE = "pipeline"
PHASE_SCHEDULER = "scheduler"
PHASE_INFERENCE = "inference"
PHASE_ROLLOUT = "rollout"
PHASE_TRAJECTORY = "trajectory"
PHASE_CACHE = "cache"
PHASE_REWARD = "reward"
PHASE_POLICY_EVAL = "policy_eval"
PHASE_ADVANTAGE = "advantage"
PHASE_TRAINING = "training"
PHASE_WEIGHT_SYNC = "weight_sync"
PHASE_ENV = "env"


# ==============================================================================
# Batch-level Metrics Keys (timing)
# ==============================================================================

# Pipeline
TIMING_STEP_TOTAL = "timing.step_total"

# Weight Synchronization
TIMING_WEIGHT_SYNC = "timing.weight_sync"

# Rollout
TIMING_ROLLOUT = "timing.rollout"
TIMING_ROLLOUT_GET_BATCH = "timing.rollout.get_batch"

# Validation
TIMING_VALIDATION = "timing.validation"
TIMING_VALIDATION_GET_BATCH = "timing.validation.get_batch"

# Scheduler
TIMING_SCHEDULER_SHRINK = "timing.scheduler.shrink"

# Reward
TIMING_REWARD_REF_LOGPROB = "timing.reward.ref_logprob"
TIMING_REWARD_RESPONSE_MASK = "timing.reward.response_mask"
TIMING_REWARD_NORMALIZE = "timing.reward.normalize"
TIMING_REWARD_TOKEN_LEVEL = "timing.reward.token_level"

# Policy Evaluation
TIMING_POLICY_EVAL_LOGPROB = "timing.policy_eval.logprob"

# Advantage
TIMING_ADVANTAGE = "timing.advantage"

# Training
TIMING_TRAINING = "timing.training"

# Other
TIMING_METRICS_COMPUTE = "timing.metrics_compute"
TIMING_LOGGING = "timing.logging"
TIMING_STOP_SERVER = "timing.stop_server"
TIMING_GENERATE = "timing.generate"


# ==============================================================================
# System/Throughput Metrics Keys
# ==============================================================================

THROUGHPUT_TOKENS_PER_SECOND = "throughput.tokens_per_second"
THROUGHPUT_TOTAL_SAMPLES = "throughput.total_samples"


# ==============================================================================
# Batch Metrics Keys
# ==============================================================================

BATCH_ADD_COUNT = "batch.add_count"
BATCH_REMOVE_COUNT = "batch.remove_count"


# ==============================================================================
# Trajectory-level Metrics Keys (per-trajectory statistics)
# ==============================================================================

# Total time
TRAJECTORY_TOTAL_TIME = "trajectory.total_time"

# Reset time
TRAJECTORY_RESET_TIME = "trajectory.reset_time"

# Step time
TRAJECTORY_STEP_TIME_MEAN = "trajectory.step_time.mean"
TRAJECTORY_STEP_TIME_MIN = "trajectory.step_time.min"
TRAJECTORY_STEP_TIME_MAX = "trajectory.step_time.max"

# Generation time
TRAJECTORY_GENERATION_TIME_MEAN = "trajectory.generation_time.mean"
TRAJECTORY_GENERATION_TIME_MIN = "trajectory.generation_time.min"
TRAJECTORY_GENERATION_TIME_MAX = "trajectory.generation_time.max"
TRAJECTORY_GENERATION_TIME_TOTAL = "trajectory.generation_time.total"

# Throughput
TRAJECTORY_THROUGHPUT_MEAN = "trajectory.throughput.mean"
TRAJECTORY_THROUGHPUT_MIN = "trajectory.throughput.min"
TRAJECTORY_THROUGHPUT_MAX = "trajectory.throughput.max"

# Response length
TRAJECTORY_RESPONSE_LENGTH = "trajectory.response_length"

# Prompt length
TRAJECTORY_PROMPT_LENGTH_MEAN = "trajectory.prompt_length.mean"
TRAJECTORY_PROMPT_LENGTH_MIN = "trajectory.prompt_length.min"
TRAJECTORY_PROMPT_LENGTH_MAX = "trajectory.prompt_length.max"

# Response length per step
TRAJECTORY_RESPONSE_LENGTH_PER_STEP_MEAN = "trajectory.response_length_per_step.mean"
TRAJECTORY_RESPONSE_LENGTH_PER_STEP_MIN = "trajectory.response_length_per_step.min"
TRAJECTORY_RESPONSE_LENGTH_PER_STEP_MAX = "trajectory.response_length_per_step.max"


# ==============================================================================
# Stage Group Definitions (for HTML visualization)
# ==============================================================================

STAGE_GROUP_PIPELINE = "Pipeline"
STAGE_GROUP_SCHEDULING = "Scheduling"
STAGE_GROUP_INFERENCE = "Inference"
STAGE_GROUP_INTERACTION = "Interaction"
STAGE_GROUP_EVALUATION = "Evaluation"
STAGE_GROUP_OPTIMIZATION = "Optimization"
STAGE_GROUP_DIAGNOSTICS = "Diagnostics"
