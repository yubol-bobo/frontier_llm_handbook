# 把训练研发连接起来的五个产物契约

日期：2026-09-08。性质：固定源码与一手报告支持的工程综合；未联合运行这些仓库。入口：[训练全流程](../../handbook/00-end-to-end.md)。

课程中的同一条联系由 [首课](../../lessons/01-one-token-to-update.md) 引入：有效 token loss（M02）→数据/mask（M03）→分布式归约（M05）→RL轨迹目标（M11/M13）。[实验 002](../../experiments/002-token-weighted-loss/README.md) 验证了分组加权的 CPU 数学反例；没有据此新增软件集成或 GPU 运行证据。学习先修与软件关系分别见 [ROADMAP](../../ROADMAP.md) 和关系总表。

仓库关系可以按 imports 画，也可以按“下一层需要上一层交付什么”画。下表属于后者：相同问题或可借鉴契约不等于已经存在的 adapter。

| 交接 | 至少保留的身份 / 语义 | 为什么影响正确性 | 证据入口 |
|---|---|---|---|
| 原始数据 → 训练 batch | 来源、处理 / tokenizer 版本、shard、sample 位置、packing / masks、实际混合 | 改了文档边界或 loss 位置，就改变了目标；只留数据集名字不足以复现 | [数据设计](../../handbook/01-data-model-pretraining-design.md)、[Marin](../repositories/marin.md)、[OLMo-core](../repositories/olmo-core.md) |
| 研究配方 → 分布式 update | 架构、初始化、优化器、token horizon、并行组、有效 token 分母 | 独立样本数不能重复乘 CP / TP / EP；局部平均不能随意再次平均 | [分布式训练](../../handbook/02-distributed-pretraining-operations.md)、[TorchTitan](../repositories/torchtitan.md)、[Megatron](../repositories/megatron-lm.md) |
| update → checkpoint → export | 参数和优化器所有权、master / 随机 / 数据状态、完整提交条件、导出布局 | 能加载推理权重不等于能续训；能续训不等于不同拓扑逐位一致 | [恢复契约](../../handbook/02-distributed-pretraining-operations.md)、[Marin 实际案例](../../handbook/04-marin-535b-live-case-study.md) |
| harness / environment → learner | 观察与动作 token、logprob、mask、分支、环境 / verifier / policy 版本 | 重建文本可能失去原始采样 token；reward 维度及其聚合会改变学习目标 | [RL 详章](../../handbook/03-posttraining-agent-rl-evaluation.md)、[RL 连接](rl.md)、[奖励实验](../../experiments/001-harbor-reward-contract/README.md) |
| learner → rollout / deployment | 完整权重版本、tokenizer、template、路由 / 精度 / sampling、缓存版本策略 | 新旧版本混合、概率不一致或旧 KV 复用可能改变行为；需要分别诊断 | [Miles](../repositories/miles.md)、[SGLang](../repositories/sglang.md)、[GPU 连接](infra.md) |

这些契约共同构成实验的可追溯性：数据与运行的身份贯穿预训练，任务与行为策略的身份贯穿 RL。Marin 的 artifact DAG 与 Pi 的会话日志都处理身份和历史，但记录对象不同；本文没有据此宣称两者已经集成。

## 一个贯穿多层的取舍：长上下文 × MoE

长上下文先改变样本长度及文档组织，随后改变 batch 中独立序列的数量、专家收到的 token 分布、显存与通信。在后训练中它还改变 rollout 耗时、KV cache、队列陈旧度和失败恢复成本。于是“扩展上下文窗口”需要数据、训练、通信、推理与环境共同验收，不能只改位置编码参数。

源码 / 配方证据：[SmolLM 阶段配置](../repositories/smollm.md)、[Marin 535B 决策链](../../handbook/04-marin-535b-live-case-study.md)、[MoE 通信](infra.md)、[异步 RL](rl.md)。它们是互补的具体实例，不能把一个实现的数值阈值套给所有项目。

## 验证顺序

先验证单个交接的身份、shape、mask 和数值，再验证跨进程 / 设备的同等语义，最后加入异步、故障与性能压力。可运行的组合依赖各 recipe 实际固定的版本，而非本实验室独立 clones 的 HEAD；见 [版本连接表](../../REPO_RELATIONSHIPS.md#版本连接表)。当前新增的是证据结构，没有新增 GPU 或完整 RL 实验结果。
