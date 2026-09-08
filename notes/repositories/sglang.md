# SGLang：缓存、调度和 MoE 后端怎样共同决定 rollout 成本

日期：2026-09-08  
阶段：L1 源码初读  
源码：`https://github.com/sgl-project/sglang` @ `30e7a3072d3f1e9bd70cd5e44146ca27c80522c4`  
验证范围：实际阅读本地 scheduler 两种循环、批次合并片段、radix 匹配/驱逐/锁引用、DeepEP v2 adapter、DeepGEMM wrapper；未启动服务、安装依赖或运行 GPU 实验。

## 核心问题

为什么 rollout 引擎不仅是 `model.generate()` 的 HTTP 包装？关键在于：在多个持续生成的请求之间复用 KV，同时安排下一批 GPU 工作、处理上一批结果，还要保证分布式 MoE 的数据布局与同步边界一致。

## 源码路径与执行链路

本地路径相对于学习库根目录。

| 证据 | 本地实现 | 固定源码 |
|---|---|---|
| S1 | `sources/sglang/python/sglang/srt/managers/scheduler.py` | [普通/重叠循环，1893–2022 行](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/managers/scheduler.py#L1893-L2022) |
| S2 | `sources/sglang/python/sglang/srt/mem_cache/radix_cache.py` | [匹配、缓存生命周期、驱逐与引用，400–680 行](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/radix_cache.py#L400-L680) |
| S3 | `sources/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py` | [可选导入、buffer 与 dispatch/combine，38–418 行](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py#L38-L418) |
| S4 | `sources/sglang/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` | [可选导入与 masked GEMM 调用，18–100 行](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py#L18-L100) |

普通循环：`ingest_requests()` → `get_next_batch_to_run()` → `run_batch()` → `process_batch_result()`。重叠循环先安排当前 forward，把结果放入 `result_queue`，再处理上一批结果；采样中依赖上一批状态的部分放在结果处理之后。

额外读到 `scheduler.py:3500–3609`：完成的 prefill 批次会合并进 `running_batch`，尚未完成的 chunked request 被排除并视情况缓存已有 KV；然后构造新的 prefill 计划。缓存侧 `match_prefix()` 返回 KV pool 的索引和树节点；活动请求持有节点引用，释放后才重新成为可驱逐容量。

## 已确认的机制与取舍

- **源码事实：前缀键不只有文本。** `RadixKey` 的 token IDs 与 `extra_key` 共同决定命名空间；不同 adapter 等上下文不应误用同一 KV。匹配还会按 page size 对齐，并可能切分树节点。
- **源码事实：命中率不是唯一状态。** `inc_lock_ref/dec_lock_ref` 沿祖先路径维护 protected/evictable token 数；`evict()` 从可驱逐叶子的优先队列释放 KV segment。缓存容量必须同时服务正在执行的请求和可复用的历史。
- **源码事实：重叠存在延迟取舍。** 连续 prefill 的特殊分支可禁用 overlap，源码注释解释其改善前一批 TTFT、可能损失吞吐。在 DP attention 下使用同步后的批次信息来统一决策，避免各 rank 走不同同步路径。
- **源码事实：MoE adapter 处理的是数据契约。** DeepEP v2 路径在 decode 用 expanded/masked 布局以避免 CPU 同步，extend 用 contiguous；检查 hidden size、top-k、每 rank token 容量；空 rank 在需要 CPU 同步时填一个零权重 dummy token。dispatch handle 必须被 combine 消耗，失败时也释放。
- **阅读推断：** 多轮 agent 请求常有共享前缀，因此 KV 与调度会影响采样成本；但效果还取决于截断/重写上下文、模型、请求分布和并发，不能由代码存在推得固定加速倍数。

## 与其他项目的连接

- **可选后端：SGLang → [DeepEP](deepep.md)。** S3 显式导入 `ElasticBuffer`，缺少库会留下错误并在选择该路径时抛出；是实质 adapter，不是 README 中的一条兼容声明。注意本次未安装，未证明两个 HEAD 的 ABI 兼容。
- **可选后端：SGLang → [DeepGEMM](deepgemm.md)。** S4 在 `ENABLE_JIT_DEEPGEMM` 下导入并调用 GEMM。另读 `deep_gemm_wrapper/configurer.py:18–36`，开关同时检查设备、可导入性和环境配置。
- **概念对应：SGLang ↔ [Harbor](harbor.md)。** 一个管理生成/缓存/算子，一个管理任务环境/验证。训练框架中的具体连接应到 [RL 笔记](../connections/rl.md) 核查，不能将三者画成无条件直接依赖链。

## 动手实验

**状态：待执行。** 选能完整放入单张 GPU 的模型，生成 token 长度一致的两组请求：A 共享长 token 前缀；B 长度一致但前缀随机不同。再分别启停 radix cache，形成四组。

控制变量：同一模型/权重、分词结果、最大输出长度、采样设置、并发度和硬件；区分冷缓存与预热后测量。观测：TTFT p50/p95、每秒输出 token、端到端耗时、缓存命中 token、KV 使用量。预期：A 的收益更明显；是否兑现要看实测。实际结果：未执行。算力：单 GPU 足够做此实验；MoE 跨机通信不在本实验范围。

## 下一步与疑问

- [ ] 从 `get_new_batch_prefill()` 追到 PrefillAdder，画出 token budget 与 chunk size 的实际约束。
- [ ] 用一个共享前缀例子手工画出 radix split 与 lock_ref 的变化。
- [ ] 核查 RL 权重更新时缓存失效、暂停/恢复的实现，避免把 serving 正确性直接套到训练 rollout。
