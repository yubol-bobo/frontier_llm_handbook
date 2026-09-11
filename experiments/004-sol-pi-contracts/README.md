# 实验 004：直接检验 SoL-Pi 的上下文压缩决策

记录日期：2026-09-11。目标是把“压缩会省钱”变成可计算、可反驳的条件。这个 CPU 实验直接调用固定上游源码里的 `decideCompaction`，不调用模型、不安装 Pi CLI、不触发真实会话压缩。结果只支持所列决策场景。

## 1. 先预测，再执行

前置阅读：[SoL-Pi 课程](../../handbook/06-sol-pi-efficient-harnesses.md)和[源码笔记](../../notes/repositories/sol-pi.md)。需要 Git、Python 和本次验证使用的 Node 22.19.0；Node 通过 `--experimental-strip-types` 读取上游 TypeScript。没有 GPU 或模型 API 要求。

从资料库根目录执行；已有锁定源码时跳过第一条：

```powershell
python tools/clone_repos.py --group harness --restore-lock
node --experimental-strip-types experiments/004-sol-pi-contracts/check.mjs
```

第一条会恢复整个 harness 分组的锁定副本，并非只获取 SoL-Pi。工具不会重置已有副本；已有 HEAD 不匹配时会报错并保留原副本。检查脚本要求 SoL-Pi HEAD 精确匹配 `d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0`，且跟踪文件没有修改。该脚本只依赖 Node 内置模块与上游纯函数，不需要在 `sources/sol-pi` 安装 npm 依赖。

执行实现：[check.mjs](check.mjs)。已保存实际输入、决策字段与执行环境：[results.json](results.json)。再次执行会刷新此结果文件及 UTC 记录日期。

## 2. 共同输入与独立算术

基础输入为 `W=80000, A=60000, M=1000, r=12.5, D=0`；当前上下文 80000，窗口 200000，平均正增长 2000；完成边界间请求数 `[4,6,5]`，剩余 4 个边界，初始压缩次数为零。r 是示例配置，不是当前 API 价格。

```text
S = A - M = 59000
incremental_cache_cost = W * max(0, r - 1) = 920000
breakeven_requests = 920000 / 59000 ≈ 15.59322
predicted_requests = 1 + floor(5 * 4) = 21
window_request_cap = floor((200000 - 80000) / 2000) = 60
first_effective_horizon = min(2 * 21, 60) = 42
```

脚本既断言上游函数的决定与原因，也独立核对这些算术及携带债务的比率。它没有复制一份上游算法来代替调用原函数。

| 场景 | 相对基础输入的变化 | 实际决定 | 原因 |
| --- | --- | --- | --- |
| first | 无 | 压缩 | `economic` |
| later_margin | 已压缩 1 次 | 延后 | `deferred_subsequent_margin` |
| later_longer_horizon | 已压缩 1 次，剩余 6 个边界 | 压缩 | `economic` |
| unpaid_debt | 上一行加债务 2000000 | 延后 | `deferred_carried_debt` |
| window_pressure | W/当前上下文 190000，A=150000，r=100 | 压缩 | `window_protection` |
| no_saving | A=M=1000 | 延后 | `non_positive_saving` |
| no_history | 完成边界样本为 null | 延后 | `horizon_unavailable` |
| no_price_ratio | r=null | 延后 | `cache_ratio_unavailable` |

**实际结果：8/8 场景通过，独立算术断言通过。** 后续压缩的 margin 为 `1.5 × 15.59322≈23.39`，所以 21 个预测请求不足；把剩余边界增至 6 后预测为 31，margin 才通过。债务场景的综合盈亏平衡请求数约 49.49，31 仍不足。

## 3. 上游完整检查的真实结果

另外在资料库外建立同 SHA 的隔离副本，使用 Windows NT 10.0.26200、Node 22.19.0、npm 10.9.3 与锁定的 Pi 0.84.2 开发依赖，执行以下命令。它们是已执行记录，不是这个无依赖 CPU 脚本的安装步骤。

```powershell
npm ci --ignore-scripts
npm run check
npm pack --dry-run --json
node scripts/check-pi-compat.mjs
```

`npm ci` 成功；`npm run check` **退出码为 1**：TypeScript 通过，Vitest 的 139 项测试中 134 通过、5 失败、0 跳过，18 个文件中 15 通过。组合命令中的打包阶段因前面的失败未执行；随后独立 dry run 成功，35 个包条目、9 项等价内容断言通过。只读 Pi API 探测也通过。[机器可读记录](upstream-validation.json)。

| 失败数量 | 测试位置 | 实际观察与局限 |
| --- | --- | --- |
| 2 | `tests/package.test.ts:15–19` | 无 shell 的 `spawnSync("npm", …)` 在此 Windows 环境报 `ENOENT`；另行执行打包成功不改变原测试失败结果。 |
| 2 | `tests/observation-pack.test.ts:254,292` | 创建符号链接 fixture 时 `EPERM`；未运行到预期保护断言。 |
| 1 | `tests/evidence-preserving-reducer.test.ts:296` | 文件模式读回 `0666`，不满足 `0600` 断言；不能据此证明 Windows ACL 隐私保护。 |

两个副本的跟踪源码均保持干净。没有通过修改测试、跳过失败或调整系统权限让结果变绿，也尚未在 Linux 重跑。上游套件使用模拟提供方与若干真实临时文件操作；真实 `AgentSession` 配合确定性摘要验证生命周期，不能证明实际模型摘要质量。

## 4. 学会解释失败，再扩展实验

- 将 `remainingBoundaries` 从 4 改为 6，先写预测，再在独立实验脚本中调用函数；解释为何“相同长度的上下文”可能作出不同决策。
- 保留窗口充足条件，把 r 设为 null；再增加窗口压力，解释经济模式不可用与窗口保护之间的关系。纯函数通过不意味着会话一定满足成功边界与原生 cut-point 条件。
- 提出一个冷缓存或任务提前结束的反例：为什么函数的 `compact=true` 不足以证明账单降低？列出需要测量的真实读写 token、摘要费用和最终任务结果。
- 对照源码中的取消、归档异常和证据遗漏行为，设计下一组 fixture。只把已运行的新场景记为执行证据，不把练习题记为完成。

交付物应包括自己的输入表、预期与实际决定、失败解释，以及明确的适用边界。前两项有部分条件重现已执行的场景；null 比率叠加窗口压力，以及取消、归档与证据遗漏的新 fixture 尚未执行。本次没有质量评测、在线 compaction、缓存命中、速度或费用复现。
