# slime：多轮 agent 的文本记录怎样成为可训练的 token 轨迹？

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/THUDM/slime @ `4c193f1f37509cca70f0e88807a9305b70f63f4e`  
验证范围：阅读本地固定快照的实现、相关配置与声明；未安装上游依赖，未启动模型、沙箱或训练。本文的控制流判断来自静态阅读，不代表复现了作者的性能或 benchmark。

## 核心问题

agent 会重写消息、压缩上下文、调用子 agent。训练不能只把最终聊天文本重新 tokenize：loss 和 rollout logprob 必须对应模型当时真正采样的 token。这个快照如何区分“作为上下文保留”和“允许反向传播”？

## 源码路径与执行链路

- [基础训练闭环与资源切换](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/train.py#L9-L94)；本地：[`train.py`](../../sources/slime/train.py)。
- [轨迹重对齐、分叉、共享前缀去重与导出](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/trajectory.py#L141-L501)；本地：[`slime/agent/trajectory.py`](../../sources/slime/slime/agent/trajectory.py)。
- [SGLang 服务导入和进程启动](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/sglang_utils/sglang_engine.py#L1-L64)；本地：[`slime/backends/sglang_utils/sglang_engine.py`](../../sources/slime/slime/backends/sglang_utils/sglang_engine.py)。
- [Megatron、loss 与 weight updater 的连接](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/actor.py#L8-L49)；本地：[`slime/backends/megatron_utils/actor.py`](../../sources/slime/slime/backends/megatron_utils/actor.py)。

基础系统链路是 `train.py` 创建 placement groups → 创建带 SGLang engines 的 rollout manager → 创建 actor/critic → 初次同步权重 → `generate` → `async_train` → 保存/释放资源 → 再同步权重。此文件逐 rollout 等待；其他 fully async 入口未在本次展开，不能从框架宣传反推此入口没有同步点。

轨迹链路另有清楚边界：adapter 提交 `TurnRecord(prompt_ids, output_ids, output_log_probs, finish_reason)` → `record_turn()` 按消息挂入 session tree → `get_trajectory()` 遍历 root-to-leaf chain → `_split_chain_into_builders()` 按 token 前缀检查 → `_SampleBuilder.to_sample()` 导出 `Sample`。

消息树判断的是消息对应关系，token builder 判断的是精确 token 连续性，两者不能混成一种“对话合并”。初始 prompt 在完整 tokens 中保留，`loss_mask` 与 `rollout_log_probs` 只导出 response region；这两个数组的长度不能错误地拿去与完整 token 长度直接比较。

## 已确认的机制与取舍

- **实现事实：** CLEAN 时仅追加 prompt 新后缀（mask=0）和新输出（mask=1）。REALIGN 会把最近 response 区域按新 prompt 替换成 mask=0；FORK 则开新 builder。分类条件取决于分歧位置与 `fork_threshold`，具体判断中比较的是新 `output_ids` 长度，不能只照“短漂移”描述自行换成 drift 长度。
- **实现事实：** `_split_chain_into_builders():469–476` 用 `response_trained` 去重：共享生成前缀在首个 leaf 训练，后续 leaf 只作为 mask=0 上下文。`get_trajectory():339–340` 给每个导出 Sample **完整 outcome reward**。本次快照不是简单 reward/K；因此旧网页或注释中的“reward split”不能替代当前实现。
- **实现事实：** REALIGN 放弃部分训练信号以维护采样来源的可信度；FORK 增加训练样本数。无可训练 token 的 builder 被过滤。session 在导出后删除，第二次调用不能视为无副作用读取。
- **阅读判断：** 真正的训练数据格式不是 `(prompt, answer, reward)` 三元组，而包括 token 来源、logprob、mask、分组、分支/样本身份和停止原因。这个边界是 harness 工程与 RL 正确性相交的位置。

额外实际阅读：`tests/test_agent/test_trajectory_manager_branching.py:1–100` 的测试组织和 token vocabulary。它提供 public API 驱动与人类可读 dump 的入口，但本次没有执行测试，测试文件存在不等于当前快照全部通过。

## 与其他项目的连接

| 对象 | 关系类型 | 本次证据与边界 |
|---|---|---|
| SGLang | 直接后端集成 | `sglang_engine.py:9,48–53` 导入并启动实际服务；本次未启动。 |
| Megatron-LM | 训练后端的直接依赖 | `actor.py:11` 导入 Core，47–49 连接 train 与 weight updater。 |
| Miles | 项目谱系 | Miles 自身 README 声明 fork 自 slime；这里没有把 Miles 当 slime 的运行依赖。具体证据汇总见 `../connections/rl.md`。 |
| APEX recipe | 概念对应 | 两者都保持 sampled token / loss mask / logprob；APEX 追求严格追加 token 状态，本文件还处理消息重写后的分叉/重对齐。没有直接依赖关系证据。 |

## 动手实验

**待执行：共享前缀与 token 漂移矩阵。** 用固定的小整数 token 构造四种两轮对话：完全前缀延续、最近 response 内漂移、早期 prompt 漂移、两个 leaf 共享 response。只改变 prefix/token 关系，reward 固定为 1；观测 Sample 数、每 token mask、同一生成片段的训练次数与每 Sample reward。预期共享片段不重复训练，失去采样来源的区域被 mask，分叉不会自动按样本数平分 reward。先适配现有 branching 测试的 public API，CPU 即可；依赖解析与 pytest 环境待准备。本次实际结果：**未执行**。

## 下一步与疑问

- [ ] 追 adapter 如何取得 SGLang token/logprob，确认采样参数与 logprob 的概率空间。
- [ ] 追 `rollout_id` / `group_index` 到 advantage 与 loss reducer，核实分叉样本的实际归一化方式。
- [ ] 比较 fully async 入口的 weight version 与数据队列；本篇不声称覆盖该路径。
