# 内容建设与维护者学习进度

更新：2026-09-08。项目处于第一轮学习，尚未完成整个领域或任何完整大模型训练复现。

本页记录维护者实际工作。读者从 [ROADMAP](ROADMAP.md) 进入课程，并使用 [个人学习模板](templates/learner-progress.md)；课程设计完成不表示个人结业。知识覆盖成熟度见 [COVERAGE](COVERAGE.md)。

已按用户明确的定位，将“前沿 LLM 端到端训练知识”与“工程能力的基础与提升”统一到仓库入口、能力层级、学习方法和验收模板。本次定位完善属于文档建设，实验与模块完成状态保持原记录。

主仓库：[yubol-bobo/frontier_llm_handbook](https://github.com/yubol-bobo/frontier_llm_handbook)，后续以 `origin/main` 为主线。本地学习目录保留 `frontier-llm-lab`；upstream clones 由清单与固定 SHA 恢复。

## 当前状态

| 项目 | 已完成 | 未完成 |
|---|---|---|
| 学习仓库 | 本地 Git、GitHub 主仓库、学习清单、方法与模板 | 后续学习持续更新此主仓库 |
| 交互网站 | 课程阅读、搜索、知识关联、数学互动实验、本地笔记与备份，GitHub Pages 发布配置 | 更多教学单元、真实模型实验与学习者反馈 |
| 源码获取 | 19/19 shallow clones，固定 SHA，源码索引 | 完整历史、submodules、LFS 大文件、权重与数据未下载 |
| 源码阅读 | 19/19 项目完成至少一条实现/配置链路初读 | 全仓库阅读与运行验证 |
| 跨项目关系 | 5 份连接专题、关系总表、概念知识树 v0.3 | 各依赖组合的实际安装兼容性 |
| 全流程研究 | 总章 + 数据设计、分布式运行、后训练详章 + Marin 535B 案例 | 完整训练执行、独立复现作者成绩与闭源配方 |
| 公开课程 | M00–M15 共 16 核心模块、32 项练习设计，F01–F06 研读设计 | 各模块更多完整 lesson、习题答案和实际执行 |
| 入门教学 | 首课：概率、手写梯度、mask、全局 token 平均 | 自动微分与完整 decoder 的教学实验 |
| 实验 | 2 个本地 CPU 实验，脚本与结果保存 | 模型 API、Docker、GPU、完整 RL 训练 |

## 逐仓库进度

`L2` 和 `L3` 均限于列出的局部问题。详细证据、待实验与疑问保存在各篇笔记。

| 顺序 | 项目 | 当前阅读层级 | 实验 |
|---:|---|---|---|
| 1 | [Pi](notes/repositories/pi.md) | L2：低层循环、工具排序、steering、转换边界 | 受控事件 trace 待执行 |
| 2 | [DeepSeek Harness](notes/repositories/deepseek-harness.md) | L1：输入→日志→请求与 invariant | keyless replay 待执行 |
| 3 | [Harbor Cookbook](notes/repositories/harbor-cookbook.md) | L2：任务、评分、示例接口 | L3 局部：Python 多维奖励断言已执行 |
| 4 | [Harbor](notes/repositories/harbor.md) | L1：Trial / verifier / 生命周期 | Docker Trial 待执行 |
| 5 | [Verifiers](notes/repositories/verifiers.md) | L1：v1 rollout、Task、Pi/Harbor adapter | 在线/离线评分对照待执行 |
| 6 | [Prime RL](notes/repositories/prime-rl.md) | L1：编排、trace→batch、GRPO | 小模型闭环待执行 |
| 7 | [APEX recipe](notes/repositories/apex-agents-skyrl-recipe.md) | L1：TITO、Harbor generator、async 入口 | 缺原始训练数据；替代任务实验待执行 |
| 8 | [SmolLM](notes/repositories/smollm.md) | L1：阶段 YAML、数据与 SFT 示例 | 已算 token budget；训练实验待执行 |
| 9 | [OLMo-core](notes/repositories/olmo-core.md) | L1：正式配方、数据、train module | 小模型机制实验待执行 |
| 10 | [Open Instruct](notes/repositories/open-instruct.md) | L1：ratio、rho、mask、同步 | 张量级 loss 实验待执行 |
| 11 | [Marin](notes/repositories/marin.md) | L2 局部：artifact DAG、535B 配置与数据/通信/恢复的交接 | 静态预算核算；运行与 DAG/cache 实验待执行 |
| 12 | [TorchTitan](notes/repositories/torchtitan.md) | L1：训练步与声明式 sharding | 分片/吞吐实验待执行 |
| 13 | [Megatron-LM](notes/repositories/megatron-lm.md) | L1：训练入口、PP、MoE dispatcher | 并行拓扑与通信实验待执行 |
| 14 | [verl](notes/repositories/verl.md) | L1：driver、advantage、Megatron engine | 数值与控制流实验待执行 |
| 15 | [slime](notes/repositories/slime.md) | L1：轨迹分支、共享前缀、训练后端 | mask/前缀实验待执行 |
| 16 | [Miles](notes/repositories/miles.md) | L1：TITO、routing replay、engine | 一致性实验待执行 |
| 17 | [SGLang](notes/repositories/sglang.md) | L1：scheduler、cache、DeepEP/GEMM 入口 | 共享前缀实验待执行 |
| 18 | [DeepGEMM](notes/repositories/deepgemm.md) | L1：JIT、布局、Mega MoE benchmark | 支持硬件上的内核实验待执行 |
| 19 | [DeepEP](notes/repositories/deepep.md) | L1：V2 Buffer、dispatch/combine、stream | 多 GPU 通信实验待执行 |

## 下一次从这里继续

**当前内容建设单元：把 M01 的 decoder 与样本路径做成下一份完整 lesson。**

首课 [一个 token 到一次更新](lessons/01-one-token-to-update.md) 已完成；实验 002 的有限差分、mask 与归一化反例通过。它只验证局部 CPU 数学，没有自动升级任何上游项目为“完整训练已验证”。

1. 按 [M01](curriculum/01-foundations-to-pretraining.md#m01) 的目标，写一份带 shape、因果 mask 和失败例子的 decoder lesson；保持普通电脑可进入的路线。
2. 从 OLMo-core 正式 recipe 选择一条数据 → loader → labels/masks → loss → optimizer → checkpoint 路径；标明每个对象的 shape、身份和 token 计数规则。
3. 在首课数值目标之上补全真实 decoder 前向/反向检查；具备隔离环境后再执行，记录实际范围与原有实验的区别。
4. 将同一个接口问题回看 Marin / Megatron / TorchTitan，区分研究配方和训练实现。

**保留的 agent RL 续学入口：Pi 的事件、状态与恢复。**

1. 核对 Pi SHA：`6160683a4a8012f0d1cd30c145df18b4ca6f5176`。
2. 从 [Pi 笔记](notes/repositories/pi.md) 的低层循环，转入 `packages/agent/src/harness/runtime/drive/generation.ts`、`runtime/reducer.ts`、session JSONL 路径。
3. 只选一个具体问题：当用户取消发生在工具已开始但结果未持久化时，下一次恢复依据什么决定重试或结束？先读调用链与已有 targeted tests。
4. 形成第二份 session 记录；具备依赖后用 faux provider 做 CPU 事件 trace，避免真实模型费用。
5. 之后转入 Harbor 的实际奖励加载，与实验 001 的 raw reward dict 结果对照。

## 需要后续核实的连接

- Prime RL 锁定 Verifiers submodule；独立 HEAD 不同，实际训练前对齐依赖。
- Verifiers Pi adapter 的 npm release 与独立 Pi HEAD 不同。
- Open Instruct 固定 OLMo-core commit，且有 4 个未下载的 LFS 测试数据文件。
- Cookbook harbor_rl 使用 Harbor feature branch；main 没有同名模块。
- APEX 需要未公开原始训练数据与指定 SkyRL checkout；完整原配方复现暂不具备材料。
- Megatron 所读 DeepEP Buffer adapter 与 DeepEP V2 主线需要明确版本关系。

以上不会阻止继续读代码或做独立 CPU 实验；它们限制的是具体端到端复现。

## 已保存的学习记录

- [交互学习网站建设](notes/sessions/2026-09-08-website.md)
- [公开课程与首课建设](notes/sessions/2026-09-08-curriculum.md)
- [超大训练全流程研究](notes/sessions/2026-09-08-end-to-end.md)
- [首次整体学习记录](notes/sessions/2026-09-08.md)
- [训练组阅读记录](notes/sessions/2026-09-08-training.md)
- [RL 组阅读记录](notes/sessions/2026-09-08-rl.md)
- [Infra 组阅读记录](notes/sessions/2026-09-08-infra.md)
- [实验 001](experiments/001-harbor-reward-contract/README.md)
- [实验 002](experiments/002-token-weighted-loss/README.md)

本仓库未配置定时任务或后台训练。下一次从当前单元继续，不把“计划做”更新成“已完成”。
