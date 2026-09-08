# APEX Agents SkyRL Recipe：知识工作任务怎样转成 token 级 RL 训练样本？

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe @ `8e7702f03b7464a36ab800a624fd911de0968a87`  
验证范围：阅读本地固定快照的实现、相关配置与声明；未安装上游依赖，未启动模型、沙箱或训练。本文的控制流判断来自静态阅读，不代表复现了作者的性能或 benchmark。

## 核心问题

这个 recipe 的教学价值在 glue code：Harbor trial 的文件操作、工具结果和 verifier outcome，怎样变成 SkyRL 所需的 `prompt_token_ids / response_ids / loss_masks / rollout_logprobs / rewards`？训练公式之外，失败分类和 tokenizer 边界都能改变学习信号。

## 源码路径与执行链路

- [严格追加 token 状态、工具观察与停止 token 对齐](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/apex_agents_skyrl_recipe/agents/tito.py#L93-L271)；本地：[`apex_agents_skyrl_recipe/agents/tito.py`](../../sources/apex-agents-skyrl-recipe/apex_agents_skyrl_recipe/agents/tito.py)。
- [并发 trial、token 输出、错误与超时分组处理](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/apex_agents_skyrl_recipe/tito_harbor_generator.py#L251-L635)；本地：[`apex_agents_skyrl_recipe/tito_harbor_generator.py`](../../sources/apex-agents-skyrl-recipe/apex_agents_skyrl_recipe/tito_harbor_generator.py)。
- [直接接入 SkyRL fully async trainer](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/apex_agents_skyrl_recipe/entrypoints/main_tito_harbor_fully_async.py#L15-L54)；本地：[`apex_agents_skyrl_recipe/entrypoints/main_tito_harbor_fully_async.py`](../../sources/apex-agents-skyrl-recipe/apex_agents_skyrl_recipe/entrypoints/main_tito_harbor_fully_async.py)。
- [Harbor 版本与 SkyRL 本地路径依赖](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/pyproject.toml#L1-L122)；本地：[`pyproject.toml`](../../sources/apex-agents-skyrl-recipe/pyproject.toml)。

`TITOHarborFullyAsyncExp.get_trainer()` 返回外部 SkyRL 的 `FullyAsyncRayPPOTrainer`；recipe 提供 `TITOHarborGenerator`。generator 为每个输入 task path 和 trajectory id 并发运行 trial，task 数可并发但实际启动受 rate limiter 控制。`_run_tito_trial()` 配置 session/cache salt、运行 Harbor Trial，读 verifier reward 与 agent metadata，再转换成 `GeneratorOutput`。

agent 侧 `TITOAgentState` 保持三条平行数组。`record_step()` 先保存本轮输入 token snapshot，再直接追加采样 token、mask=1 和真实 logprob；工具观察通过固定 dummy base 的 chat template delta 生成，追加 mask=0 和 logprob placeholder。stop token 的重叠要按 GLM/ChatML 等模板边界处理，避免重复 token 或丢换行。

generator 以首个 mask=1 位置切分初始 prompt 与 response region；response 中仍可包含 mask=0 的工具/环境 token。因此 `response_ids` 并不等于“每个 token 都由模型产生”。

## 已确认的机制与取舍

- **实现事实：** fixed-base tokenization 不重新 tokenize 全部历史，减少随轮数增长的历史处理；注释的 O(1) 应理解为相对历史轮数，不是对工具观察文本长度也常数。这个方法依赖模板约定，不能未经测试推广到任意 tokenizer。
- **实现事实：** cache salt 当前实现取 `weight_version` 并 assert 存在，返回 `weight_version=...`；不是 global_step，也没有按 docstring 所说将模型名拼进字符串。源码比注释更具体。异步情况下 policy version 与 step 不一定同步。
- **实现事实：** per-turn `length` 截断强制 reward=0，并优先于 context-length；后者在开启 overlong filtering 时将 loss mask 清零。超时、环境错误和“任务做错”不是同一种负样本。
- **实现事实：** 同一个 instance 的任一非 timeout error 会 mask 整组；timeout 默认达到阈值 2 时 mask 整组，否则仅 mask 超时成员。策略按 `trajectory_id.instance_id` 分组，不是按返回时间。清零错误不能被描述成“让模型学习环境故障”。
- **实现事实：** session slot 在 finally 中释放；不同 retry 使用新 session id。e2e 计时排除了首次 rate-limiter 等待，但包括 retry 间耗时，所以不能直接与含排队时延指标混比。
- **阅读判断：** 这些局部策略决定有效训练数据的选择偏差；复现 reward 曲线前应先复现错误/超时/有效 token 的统计口径。

## 与其他项目的连接

| 对象 | 关系类型 | 本次证据与边界 |
|---|---|---|
| SkyRL | 直接外部依赖 | entrypoint:16,33 导入/创建 trainer；pyproject:14,116–117 指向 SkyRL 与 skyrl-gym。SkyRL **不在本次 19 个独立源码 clone 范围内**。 |
| Harbor | 直接依赖 | pyproject:15 锁 `harbor[modal]==0.21.0`；generator:29–30 导入 TrialConfig/Trial。旁边 `sources/harbor` 的最新 HEAD 不自动匹配此版本。 |
| Megatron-LM | 经 SkyRL extra 的后端依赖 | `skyrl[megatron]` 明确请求；本次没有本地 SkyRL 实现，不能声称审计了 Megatron 执行链。 |
| slime | 概念对应 | 同样保持真实 sampled token/logprob 与观察 mask；slime 还容纳消息重写后的分叉/对齐。本次没有直接软件依赖证据。 |

**运行准备状态：** pyproject 中 SkyRL 是作者机器路径 `/mnt/local_storage/oss/SkyRL-v0.3.0`，注释对应 0.3.0 release commit 前缀 `b8a5caaa`。当前 Windows 学习库没有该目录；没有安装 CUDA 依赖、认证、镜像或任务数据。本次完成源代码阅读，不宣称 recipe 已可直接运行。

## 动手实验

**待执行：从失败类型到训练 mask。** 合成同 instance 的四个 `TITOHarborAgentOutput`，固定 response tokens/reward，依次设置 0/1/2 个 timeout 或 1 个 error；比较整组有效 token 数、reward、各 stop_reason 与计数器。控制 timeout threshold=2、组大小=4；预期 1 个 timeout 只失效该条，2 个 timeout 或一个 error 失效整组。实际结果：**未执行**。

第二个待执行实验对 GLM 与 Qwen tokenizer 的两轮工具调用做 token 边界对照，检查停止 token、换行和三条数组长度；先用 CPU 加 tokenizer assets，不需完整训练。完整 recipe 再按作者 release 锁、Linux/CUDA 和所需 sandbox 配置准备，算力规模需选定模型后核算。

## 下一步与疑问

- [ ] 读取 `agents/llm.py`，确认 `/v1/completions` 的 token 输入、输出与 logprob 参数。
- [ ] 跟踪 `entrypoints/common.py` 的数据集构造与 trainer config；仍不把外部 SkyRL trainer 当已读。
- [ ] 补齐运行时版本/镜像/数据后再做 L3；把系统失败率与模型任务成功率分开记录。
