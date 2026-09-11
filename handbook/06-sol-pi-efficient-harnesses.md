# SoL-Pi 学习指南：在保持任务质量的前提下改进 harness 效率

日期：2026-09-11。源码基线：`NVlabs/SoL-Pi@d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0`。配合[源码笔记](../notes/repositories/sol-pi.md)与[实验 004](../experiments/004-sol-pi-contracts/README.md)学习，分别保存阅读证据、推断和执行记录。

## 1. 学习位置与公开边界

学习顺序是 **M01 → M12 的 Pi 基础 → 四项机制 → M09 缓存经济性 → M14 配对评估**。M12 的纯 harness 阅读可提前到 M01 后；这条路线不替代后训练的数学先修。目标是能解释运行过程、设计反例并审计效率收益，而不是读完就宣称掌握完整自动研究系统。

本包公开四项独立扩展，生成这些设计的私有自动提案／搜索循环没有随包发布。四项机制均默认关闭，并通过 Pi 公开 API 工作；不要从项目标题推断已获得搜索器、训练器或完整研究流程。[发布范围 L21–45](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/README.md#L21-L45)、[默认配置 L27–36](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L27-L36)。

作者从 152 个候选方向中筛出四项机制；在封存的 51 任务 EdgeBench 上，报告约保留 Pi 平均得分的 94%，同时减少 45–49% token。这是带质量取舍的作者报告，本资料库尚未复现。[官方研究说明](https://nvlabs.github.io/SoL-Pi/)。

## 2. 从 M01 进入 M12：先追一条 Pi 轨迹

先读 [Pi 笔记](../notes/repositories/pi.md)，画出用户任务、模型请求、工具执行、反馈与终止的顺序。把原始输出、持久会话记录和本次提供方可见消息分列；给每次请求编号，说明工具成功与任务完成为什么需要不同证据。若尚不能解释取消、恢复和失败返回，先完成 [M12 的基础练习](../curriculum/03-posttraining-and-agents.md#m12)。

再追 SoL-Pi 的 `session_start`：它取得受信任上下文中的配置，只初始化一次，依次注册 Action Fusion、ObservationPack、EPR 与 Online Context Compact。注册顺序与各自处理的事件共同决定组合行为，不能把四个开关当作互不影响的黑箱。[入口 L13–36](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/index.ts#L13-L36)。

## 3. 四项机制：每项都找一条正常路径和反例

按表逐项阅读。每项提交“输入 → 状态变化 → 模型看见的结果”小图，并指出保证止于哪里。先单独理解，再检查组合；可回取不等于模型一定回取，引文真实也不等于证据完整。

| 机制与源码入口 | 需要掌握的契约 | 练习反例 |
|---|---|---|
| [Action Fusion L95–125](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/then-run.ts#L95-L125) | 同文件队列覆盖修改与后续命令；修改失败则跳过命令，命令失败保留修改。 | 后续检查失败，确认文件仍已修改；两次内容散列不能充当外部写入的全局锁。 |
| [ObservationPack L137–209](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/index.ts#L137-L209) | 合格大文本前两次完整投影，之后用稳定引用；归档后才替换，原会话不就地改写。 | 将关键事实放在中段；从零沿 `next_offset` 回取，比较重组后的字节。 |
| [EPR 校验 L82–143](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts#L82-L143) | [归档并委托模型生成收据 L77–147](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/index.ts#L77-L147)；校验 schema、源散列、状态与引文，校验失败保留原输出。 | 构造引文真实却遗漏另一关键失败的收据，区分真实性和覆盖度。 |
| [Online Context Compact L289–370](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L289-L370) | 计划边界经成本与可压缩性检查后停止运行，在 `agent_settled` 调用 Pi 原生压缩。 | 比较经济性通过与原生不可压缩；成功后才[触发续行 L381–407](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L381-L407)。 |

## 4. 回访 M09：减少 token 不自动减少账单

压缩减少后续重放，却可能引入缓存重写与摘要费用。先识别缓存读写单位，再解释上游的简化决策式；它使用配置中的价格比与请求数估计，不是实际账单计算器。[经济性 L149–196](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/economics.ts#L149-L196)。

```text
savingTokens = archiveTokens - memoTokens
incrementalRatio = max(cacheWriteReadRatio - 1, 0)
breakevenRequests = writeTokens * incrementalRatio / savingTokens
combinedBreakevenRequests = (carriedDebtTokens + writeTokens * incrementalRatio) / savingTokens
```

最后两式要求 `savingTokens > 0` 且价格比可用。依次改变剩余请求、首次／后续压缩、未偿缓存成本与窗口余量，先预测 `reason` 再运行。窗口保护可越过经济门槛，但仍要求正节省；外层还会拒绝原生不可压缩的上下文。解释每个分支比记住一个“该压缩”的阈值更重要。

## 5. CPU 实验与本次已验证范围

[实验 004](../experiments/004-sol-pi-contracts/README.md)直接加载真实上游 `economics.ts`，通过 Node `--experimental-strip-types` 执行八个确定性场景。练习时保留输入、预期、实际值和断言，不另写一个等价函数冒充上游执行。以下隔离副本验证环境为 Windows、Node 22.19.0、npm 10.9.3；失败也属于结果。

| 检查 | 本次结果 | 可以支持的结论与限制 |
|---|---|---|
| 经济性函数 | 8 个场景全部 PASS | 验证这些输入对应的决策契约；未运行真实压缩。 |
| TypeScript | PASS | 当前依赖环境中的类型检查通过。 |
| 完整 Vitest 套件 | 139 例：134 PASS、5 FAIL | 2 例 `spawnSync npm ENOENT`、2 例符号链接 `EPERM`、1 例 POSIX 文件权限预期 `0600` 与实际 `0666` 不符；不能记成全绿。 |
| 包检查 | dry-run 35 个文件；9 项包断言 PASS | 验证发布清单和断言，未发布包。 |
| Pi 兼容性检查 | PASS | 限于本次检查的公开 API 与版本环境。 |
| 真实模型与成本评测 | 未执行 | 未调用在线提供方，未测真实压缩续行、任务成功率或费用收益。 |

阅读上游测试说明它打算保障什么；运行记录说明本机实际通过了什么。不要把平台错误删除后报告原套件通过，也不要把函数断言通过写成真实模型能力提升。遇到失败先保存轨迹、环境和原因，再决定是否换兼容平台复测。

## 6. M14 配对评估：先过质量门，再谈效率

下一步是真实运行计划，尚未执行。为每个任务固定模型版本、代码快照、初始文件、权限、预算与独立 verifier，比较全部关闭的基线和一次只开启一项机制的方案；单项验收后再测组合。重复运行并保留所有失败，避免只挑成功样本。

- **质量门：** 预先规定完成条件及允许退化范围，检查漏做、证据丢失、错误恢复和最终产物；评分器故障单列，不能混同任务失败。
- **成本门：** 汇总主模型、reducer、摘要、重试、回取、缓存读写的实际使用与费用，同时报告端到端延迟；质量未达标时不宣称效率改进。
- **比较记录：** 保存每对任务的质量差、费用差、请求数与失败类别，报告重复运行的不确定性；若更便宜却更慢，明确给出取舍。

## 7. 交付物与退出标准

先完成 CPU 可复查交付，再选择需要 API 或本地模型的扩展。无需 GPU 也能练习路径追踪、函数边界和评估设计；GPU 或在线调用是另一个有预算的实验阶段，不能预填其结果。

| 交付物 | 退出标准 |
|---|---|
| `reading-map.md` 与四张路径图 | 每项结论有固定源码入口；区分正常路径、异常与未覆盖边界。 |
| `contracts.json` 与原始运行日志 | 可复算八场景；至少解释一个经济拒绝和一个窗口保护决定。 |
| `failure-analysis.md` | 如实记录五项上游测试失败及其环境因素，不虚构修复或复测。 |
| `paired-eval-plan.md` | 列出任务、基线、变量、预算、质量／成本门和失败分类；运行前冻结。 |

能解释为什么一个“更短”的观察可能损害后续任务，并展示如何发现这种损害，才算完成本专题的核心学习；是否得到正向收益不是通关条件。

## 8. M13 的未来连接：轨迹与奖励仍需另建契约

沿[后训练课程](../curriculum/03-posttraining-and-agents.md#m13)与[知识树](../KNOWLEDGE_TREE.md)继续，把 harness 行为连接到轨迹和奖励。SoL-Pi 的公开包没有实现 RL 参数更新；会话文本也不是完整训练样本。接入训练前还需保留实际采样 token、logprob、loss mask、模型／策略版本、分支与 verifier 结果，并解释压缩前后的可见信息。这个连接是后续研究方向，不是本次已运行的训练闭环。
