# Agentic RL 推理调度引擎设计:基于双信标 + 分级存储的 Admission Controller

**日期:** 2026-04-27
**关联文档:** `insight/new_insight/技术设计方案：基于"双信标全维状态感知"与分级存储的 Agentic RL 推理调度引擎.md`
**RL 框架:** ROLL
**分级存储:** LMCache 0.4.1
**Controller:** 自研

---

## 1. 系统定位与边界

**定位:** 在 ROLL 的 `RolloutScheduler` 内新增 `AdmissionController` 子模块,把 Agentic RL rollout 阶段的并发管理从 open-loop overcommit 升级为 closed-loop 调度。训练端契约不变(`rollout_batch_size` 仍是稳定输入输出),所有动态行为收敛在 rollout buffer 之内。

**职责边界:**

- **进:** 采集吞吐、KV 余量、session 状态等 beacons,做出 admission 决策
- **不进:** 不修改训练算法、不修改 vLLM 内部、不替换 LMCache 策略,只调用其暴露的换入换出接口
- **`AgenticKVRuntime` 的最小扩展:** 仅新增两个查询/动作 API(`query_session_states()`, `request_swap_out(session_ids)`),不改 session 状态机
- **可灰度:** 配置开关 `admission_controller.enable`,关闭时退化为现有 `select_ready_requests` 行为

**核心边界原则:** *batch size 管训练语义,controller 管采样过程。* Controller 内部用纯数据快照接口,可独立单元测试。

**架构选择(A 为主 + D 最小扩展):**

- **A**: `AdmissionController` 作为 `RolloutScheduler` 子模块,复用其全局视图
- **D 最小扩展**: `AgenticKVRuntime` 暴露查询/动作 API,不参与决策

不采用方案 B(独立 SchedulingEngine 层)因为目前只有一个 caller,YAGNI;但通过纯数据接口保持 controller 可独立测试,未来如需独立成层是低成本重构。

---

## 2. 三层 Batch 分离与配置

### 2.1 三层语义

| 层 | 配置项 | 作用 | 谁控制 |
|---|---|---|---|
| 训练批 | `rollout_batch_size`(沿用现有) | 每轮交付给 PPO/GRPO 的样本数 | 算法侧固定 |
| 轨迹池 | `collection_pool_size` | rollout 阶段同时维护的逻辑 trajectory 数 | 配置静态设置 |
| 活跃请求 | `active_request_cap` | 当前允许进入 vLLM 的 generation request 数 | controller 动态调节 |

### 2.2 配置(精简版)

新增 `AdmissionControllerConfig` 挂在 `AgenticConfig` 下:

```yaml
admission_controller:
  enable: false                          # 灰度开关,默认关闭
  collection_pool_size: 160              # 轨迹池规模
  active_request_cap_min: 8
  active_request_cap_max: 64
  active_request_cap_init: 32
  window_seconds: 2.0                    # 窗口级反馈周期
  beacon_window_seconds: 5.0             # 吞吐滑动窗口
  emergency_kv_low_watermark: 0.10       # KV 余量紧急阈值(事件驱动)
  emergency_kv_high_watermark: 0.25      # 解除紧急的 KV 余量
  shadow_pid: true                       # 阶段 1 开启影子 PID 日志
  fallback_wait_seconds: 10.0            # tool 无历史耗时时的兜底值
```

**单一动态调节变量:** `active_request_cap`。`resume_budget` 与 `prefill_budget` 推迟到阶段 2 或更晚,只有当影子日志确认 resume/prefill storm 真实发生时才引入。

### 2.3 关系约束

- `collection_pool_size ≥ rollout_batch_size`
- `active_request_cap_max ≤ collection_pool_size`
- 训练端只看 rollout buffer 的最终 N 条样本,与池大小无关

### 2.4 池大小估算(文档化)

```
collection_pool_size ≈ active_request_cap_max / gpu_duty_cycle
gpu_duty_cycle ≈ E[GPU 推理时间] / E[GPU 推理时间 + tool 等待时间]
```

---

## 3. 双信标采集与状态分类

### 3.1 BeaconCollector(纯数据采集)

```python
@dataclass(frozen=True)
class BeaconSnapshot:
    timestamp: float
    throughput_tokens_per_sec: float    # T̄: 滑动窗口平均
    throughput_delta: float              # ΔT: 与上一窗口的差值/秒
    kv_free_ratio: float                 # M: free_blocks / total_blocks ∈ [0,1]
    inflight_requests: int
    waiting_sessions: int                # tool_wait + sleeping_in_lmcache
    ready_to_resume_sessions: int
```

**信号来源(全部已存在,无需改 vLLM/LMCache):**

- **吞吐**: 累加 `inference.metrics` span 中已有的 generated tokens,按 `beacon_window_seconds` 滑窗
- **KV 余量**: 复用现有 `FreeBlockWatermark`(`akv/watermark.py`)
- **session 状态计数**: 通过新增 `AgenticKVRuntime.query_session_states()` 聚合

### 3.2 StateClassifier(纯函数,可单测)

```python
class SystemState(Enum):
    COLD_START      = "cold_start"        # T̄ 低, ΔT 正, M 高
    PARETO          = "pareto"            # T̄ 高, ΔT≈0, M 中/低
    THRASHING       = "thrashing"         # T̄ 低, ΔT 负, M 低
    TOOL_WAIT_BUBBLE = "tool_wait_bubble" # T̄ 低, ΔT 负或 0, M 高
    UNKNOWN         = "unknown"
```

**分类阈值(模块内常量,初始基于经验,后续从影子日志校准):**

- `throughput_high_ratio: 0.7`(相对 roofline 或历史峰值)
- `throughput_delta_dead_zone: ±5%`(避免抖动)
- `kv_high_threshold: 0.4`、`kv_low_threshold: 0.15`

### 3.3 BudgetAllocator(阶段 1: 阈值规则)

```python
@dataclass(frozen=True)
class AdmissionDecision:
    target_active_cap: int
    swap_out_pressure: float  # 0=不换出, 1=最大压力换出
    reason: str
    shadow_pid_suggestion: Optional[int] = None  # 阶段 1 记录但不执行
```

**规则表:**

| 状态 | `target_active_cap` 调整 | `swap_out_pressure` |
|---|---|---|
| COLD_START | `+step_up`(如 +4) | 0 |
| PARETO | 维持 | 0 |
| THRASHING | `-step_down_aggressive`(如 -8) | 0.7 |
| TOOL_WAIT_BUBBLE | `+step_up`(如 +4) | 0.3 |
| UNKNOWN | 维持 | 0 |

所有调整 clamp 到 `[active_request_cap_min, active_request_cap_max]`。

### 3.4 紧急路径(事件驱动 + 防抖)

绕过窗口周期,立即生效:

- KV 余量跌破 `emergency_kv_low_watermark` → 立即把 `active_request_cap` 降到 `min`,触发主动换出
- 恢复到 `emergency_kv_high_watermark` 以上才解除紧急态
- 双水位形成 hysteresis,防止抖动
- 紧急态期间窗口控制器**只能减不能加**

### 3.5 影子 PID 记录器

阶段 1 同时运行 PD 控制器(无 I,避免积分饱和):

```
e(t) = throughput_target - throughput_observed
pid_suggestion = current_cap + kP*e + kD*de/dt
```

**只写入 tracing span,不参与决策。** `shadow_pid_suggestion` 与实际决策一起入 trace,后续离线分析。

---

## 4. LMCache 协同:分级驻留 + 主动换出

### 4.1 Session 优先级(基于事实 + 相对历史预测)

| 优先级 | 判定依据 | 来源 | HBM 驻留策略 |
|---|---|---|---|
| P0 | `session.state == RUNNING` | 事实(状态机) | 必须驻留,禁止换出 |
| P1 | `session.state == READY_TO_RESUME` | 事实(状态机) | 优先驻留 |
| P2 | WAIT 中,`remaining_ratio < SHORT_RATIO` | 预测(EWMA) | 可被动换出 |
| P3 | WAIT 中,`remaining_ratio ≥ SHORT_RATIO` | 预测(EWMA) | 主动换出候选 |

P0/P1 由 session 状态机直接决定,不依赖时间预测;P2/P3 由相对剩余比例决定:

```
expected_total = ewma_per_tool[tool_name]    # 历史耗时 EWMA
expected_remaining = max(0, expected_total - elapsed_wait)
remaining_ratio = expected_remaining / expected_total

P2 if remaining_ratio < SHORT_RATIO else P3
```

`SHORT_RATIO`(模块内常量,默认 0.3)是无量纲量,不暴露为配置。

**冷启动(无历史)**:直接归 P3,把 HBM 让出来给有数据的 session。`expected_total` 用 `fallback_wait_seconds` 兜底。

### 4.2 ToolDurationEstimator

- 每个 `tool_name` 维护历史耗时的 EWMA(`alpha=0.3`,模块内常量)
- 数据来源: 现有 `trajectory.tool_call` span 已有耗时,在线累积
- 不做 per-(tool, env) 细分,不做 trajectory 内修正(YAGNI)
- 仅当影子日志显示同一 tool 耗时方差极大、误判频繁时再细分

### 4.3 主动换出策略

**触发时机:**

1. 状态分类为 `TOOL_WAIT_BUBBLE` 且 `swap_out_pressure > 0`:为即将准入的新 request 提前腾空间
2. 紧急路径(KV 触低水位):立即按 P3 → P2 顺序换出,直到 KV 回到 `emergency_kv_high_watermark`
3. 预防性换出推迟到阶段 2

**换出数量:**

```
target_swap_count = ceil(swap_out_pressure * num_P3_sessions)
                    + (emergency ? num_P2_sessions : 0)
```

**换出排序:** 按 `expected_remaining`(绝对秒数)降序——预计还要等最久的先走。P2/P3 分级用相对比例(语义长短),换出排序用绝对值(物理时间价值),两者不矛盾。

### 4.4 与 LMCache 接口

- 复用现有 active-only HBM mode 已验证的 evict 能力
- 通过 `AgenticKVRuntime.request_swap_out(session_ids)` 暴露给 controller
- runtime 内部调用 vLLM strategy 的 LMCache hook(已有)
- 换入由 LMCache 自身处理: next request 带 `lmcache.tag.akv_session`,LMCache 命中即载回

### 4.5 不做的事(明确边界)

- 不实现预取(tool 返回时间不可预测,风险大于收益)
- 不替换 LMCache 内部 LRU/LFU 策略
- 不做跨 worker 的 session 迁移

---

## 5. 控制循环与代码结构

### 5.1 新增/修改文件

```
roll/pipeline/agentic/
├── agentic_config.py                    [修改] 新增 AdmissionControllerConfig
├── scheduler/
│   └── (现有 RolloutScheduler)          [修改] 注入 AdmissionController
├── admission/                           [新增目录]
│   ├── __init__.py
│   ├── config.py                        AdmissionControllerConfig dataclass
│   ├── snapshots.py                     BeaconSnapshot, PoolStateSnapshot, AdmissionDecision
│   ├── beacon_collector.py              BeaconCollector
│   ├── state_classifier.py              StateClassifier(纯函数)
│   ├── budget_allocator.py              BudgetAllocator(阈值规则版)
│   ├── tool_duration_estimator.py       ToolDurationEstimator(EWMA)
│   ├── session_priority.py              P0/P1/P2/P3 评分
│   ├── shadow_pid.py                    影子 PD 控制器
│   └── controller.py                    AdmissionController(组装)
└── akv/
    └── runtime.py                       [修改] 新增 query_session_states / request_swap_out
```

### 5.2 控制循环(集成到 RolloutScheduler)

```python
while len(rollout_buffer) < rollout_batch_size:
    # === 紧急路径:事件驱动,绕过窗口 ===
    if controller.check_emergency():
        controller.apply_emergency()         # cap→min, 主动换出 P3+P2

    # === 窗口路径:周期性反馈 ===
    if window_elapsed():
        beacons = beacon_collector.snapshot()
        pool_state = scheduler.snapshot_pool()
        decision = controller.decide(beacons, pool_state)

        controller.record_shadow_pid(beacons, decision)

        scheduler.apply_decision(decision)
        if decision.swap_out_pressure > 0:
            sessions_to_evict = priority_ranker.select_for_eviction(
                pool_state, decision.swap_out_pressure
            )
            akv_runtime.request_swap_out(sessions_to_evict)

    # === 既有逻辑:select_ready_requests 按 active_cap 限流 ===
    ready = select_ready_requests(pool, cap=scheduler.active_cap)
    dispatch_to_vllm(ready)
    update_states_from_finished()
    append_to_buffer(rollout_buffer)

return rollout_buffer[:rollout_batch_size]
```

### 5.3 AdmissionController 接口(纯数据,可单测)

```python
class AdmissionController:
    def decide(
        self,
        beacons: BeaconSnapshot,
        pool_state: PoolStateSnapshot,
    ) -> AdmissionDecision: ...

    def check_emergency(self) -> bool: ...

    def apply_emergency(self) -> AdmissionDecision: ...

    def record_shadow_pid(
        self, beacons: BeaconSnapshot, actual: AdmissionDecision
    ) -> None: ...
```

不持有 scheduler 引用,所有输入通过快照,所有输出通过 decision 对象。

### 5.4 AgenticKVRuntime 最小扩展

```python
class AgenticKVRuntime:
    # 既有方法不变

    def query_session_states(self) -> List[SessionStateRecord]:
        """返回所有 session 的 (id, state, tool_name, wait_started_at)"""

    def request_swap_out(self, session_ids: List[str]) -> None:
        """触发 LMCache evict,通过现有 vLLM strategy hook"""
```

仅这两个新方法,不改 session 状态机,不改现有调用方。

### 5.5 Tracing 集成

每次 `controller.decide()` 写一个 `admission.decision` span,attributes:

- `beacons.*`(吞吐、ΔT、KV 余量)
- `state`(分类结果)
- `decision.target_active_cap` / `swap_out_pressure`
- `shadow_pid.suggestion`
- `decision.reason`

复用现有 `roll/utils/tracing/`,无需新基础设施。

### 5.6 灰度与回滚

- `admission_controller.enable=false` 时,`AdmissionController` 不创建,scheduler 走现有 `select_ready_requests`,**零行为变化**
- 启用后,所有 controller 输出 clamp 到 `[active_request_cap_min, active_request_cap_max]`,最坏退化为固定 cap
- 影子 PID 数据从 tracing span 离线分析,不影响在线行为

---

## 6. 测试策略与阶段 2 演进

### 6.1 单元测试

| 模块 | 测试方式 |
|---|---|
| `StateClassifier` | 喂构造 `BeaconSnapshot`,断言四种状态识别 + dead_zone 行为 |
| `BudgetAllocator` | 喂 `(state, current_cap)`,断言 `target_cap` 在 min/max 内,方向正确 |
| `ToolDurationEstimator` | 喂 tool 历史耗时序列,断言 EWMA 收敛、冷启动 fallback |
| `SessionPriority` | 喂构造 wait sessions,断言 P2/P3 分级符合 `remaining_ratio` 语义 |
| `ShadowPIDRecorder` | 喂时序 beacons,断言 PD 输出与手算一致 |
| `AdmissionController.decide` | 端到端断言 emergency 优先级、影子记录被调用 |

全部不依赖 Ray、vLLM、LMCache,pytest 秒级反馈。

### 6.2 集成测试

- 用 `examples/agentic_demo/agent_val_frozen_lake_single_node_demo.yaml`
- 关键断言:
  1. `enable=false` 时行为与基线完全一致
  2. `enable=true` 时,KV 余量不会突破 `emergency_kv_low_watermark` 持续超过一个窗口
  3. tracing span `admission.decision` 数量符合预期
  4. `shadow_pid.suggestion` 字段非空

### 6.3 端到端验证

- 在 `agent_rollout_rock_swe_traced_lmcache_baseline.yaml` 上对比基线 vs 启用
- 关注指标: rollout 总耗时、GPU 利用率方差、tool wait 期间 throughput floor、KV 余量方差
- **预期**: 锯齿压平,goodput 提升,无新增 OOM

### 6.4 阶段 2 演进路径

阶段 1 落地并跑足够 episodes 后,基于影子日志做:

1. **PID 参数整定**: 用记录的 `(beacons, shadow_pid_suggestion, actual_decision, post_throughput)` 离线参数搜索
2. **状态分类阈值校准**: 检查 `throughput_high_ratio` 等阈值是否合适
3. **决定是否引入 resume/prefill budget**: 仅当影子日志显示 resume storm 真实存在时才加
4. **PID 灰度切换**: 10% → 50% → 100%,与阈值规则版 A/B

阶段 2 不再改架构,仅替换 `BudgetAllocator` 内部策略。这是 controller 纯函数接口设计的回报。

---

## 7. 自适应吞吐目标(借鉴 TCP 拥塞控制)

### 7.1 问题

影子 PID 需要 `throughput_target` 来计算误差 `e(t) = target - observed`,但 GPU 吞吐上限受多种因素动态影响(batch composition、KV cache 命中率、tool wait 分布),无法预先配置固定值。

### 7.2 方案:AIMD 风格的自适应峰值探测

借鉴 TCP 拥塞控制思路,动态探测并追踪"当前工作负载下的吞吐峰值":

```python
class AdaptiveThroughputTarget:
    def __init__(self):
        self.estimated_peak = 0.0  # 当前估计的峰值吞吐
        self.ewma_alpha = 0.1      # 衰减系数
    
    def update(self, observed_throughput, kv_free_ratio):
        # 观测到更高吞吐 + KV 有余量 → 上调峰值估计
        if observed_throughput > self.estimated_peak and kv_free_ratio > 0.2:
            self.estimated_peak = observed_throughput
        # 否则用 EWMA 缓慢衰减(类似 TCP RTT 估计)
        else:
            self.estimated_peak = (self.ewma_alpha * observed_throughput + 
                                   (1 - self.ewma_alpha) * self.estimated_peak)
    
    def get_target(self):
        return self.estimated_peak * 0.95  # 目标是峰值的 95%,留 headroom
```

**优点:**
- 自适应,无需配置
- 冷启动自然:初始 `estimated_peak=0`,首次观测自动初始化
- 工作负载变化时自动跟踪

### 7.3 可选:启动时 Warmup 探测

在 rollout 开始前,可选地运行短暂 warmup 阶段:

- 用 dummy prompts 逐步增加并发(如 8 → 16 → 32 → 64)
- 每个并发级别跑 2-3 个窗口,记录吞吐
- 初始化 `estimated_peak` 为 warmup 中观测到的最大值
- **不影响训练**:warmup 在 rollout 循环外,生成的 tokens 不进 buffer

Warmup 可选,因为 AIMD 本身能在线探测,但 warmup 能加速收敛。

### 7.4 开放问题

- `select_ready_requests` 现有签名是否已支持注入 cap 限流? 如不支持,需要在阶段 1 的 RolloutScheduler 改造中先扩展接口。
