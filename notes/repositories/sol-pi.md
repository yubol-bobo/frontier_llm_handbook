# SoL-Pi：沿四种机制学习 Agent Harness 提效

首次源码学习：2026-09-11。快照：`d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0`，版本 `0.1.0`，MIT，64 个跟踪文件。本次达到四种机制的局部 L2 阅读，并完成局部执行验证；不代表整个项目或论文成绩已经复现。[固定源码](https://github.com/NVlabs/SoL-Pi/tree/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0)。

## 阅读定位与配置边界

这是基于 Pi 公开扩展 API 发布的独立扩展包，适合接在 [Pi 事件循环学习](pi.md) 之后。它把工具动作、工具结果和上下文重组作为可修改的系统部件；当前包没有训练模型权重，也没有发布完整的自动研究提案、筛选与搜索流水线。学习顺序见[专题课程](../../handbook/06-sol-pi-efficient-harnesses.md)。

项目配置只有在允许信任当前目录时才优先使用；选中的项目文件替代用户配置文件，再与内置默认值组合，并非两个文件逐项合并。四个开关默认全部关闭，配置在会话启动注册扩展时读取。开发锁定 Pi 0.84.2；本资料库独立 Pi 快照为 0.85.1，两者不可直接视为已验证的依赖组合。[配置选择](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L48-L75)、[默认值与合并](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L27-L36)、[返回配置](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L115-L121)。

| 机制 | 改变的接口 | 希望减少的开销 | 必须检查的行为 |
| --- | --- | --- | --- |
| Action Fusion | `edit/write` + `then_run` | 修改与检查之间的模型往返 | 修改成功、检查失败后的文件与顺序 |
| ObservationPack | 发给模型的上下文投影 | 大输出的重复传输 | 原文存档、分页回取和失败回退 |
| Evidence-Preserving Reducer | 诊断工具的结果 | 重复、冗长日志 | 引用出处、遗漏的证据和原文回读 |
| Online Context Compact | 计划边界与会话生命周期 | 长上下文与缓存重建的总成本 | 经济门槛、摘要质量、取消与续跑 |

## 本次实际验证

[实验 004](../../experiments/004-sol-pi-contracts/README.md) 直接导入固定版本的 `economics.ts`，8 个确定性场景与独立算术断言全部通过。另在隔离副本运行上游检查：TypeScript 通过，Vitest **134/139 通过、5 项失败、0 跳过**；失败涉及 Windows 的 npm 子进程解析、符号链接 fixture 和 POSIX 文件模式断言。另行执行的打包检查与 Pi API 探测通过。这些结果不能改写为上游全套检查通过；具体命令、失败位置与范围保存在实验记录。

下文区分源码读到的断言、已经执行的 fixture 与尚待设计的练习。未调用真实模型，也未测量质量、账单、缓存命中或任务延迟。

## Action Fusion 与 ObservationPack：顺序执行与可回取观察

阅读基线为 SoL-Pi 提交 `d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0`。两项机制都默认关闭；入口在 `session_start` 取得配置后注册，顺序为 Action Fusion、ObservationPack、EPR、Online Context Compact。开发依赖固定为 Pi 0.84.2，运行时使用 peer dependencies；这说明集成基线，不证明任意版本兼容。[默认配置 L27–36](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L27-L36)、[注册入口 L13–36](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/index.ts#L13-L36)、[依赖 L31–47](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/package.json#L31-L47)。

### Action Fusion：沿三个入口追踪

1. [`createActionFusionExtension` L69–132](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/index.ts#L69-L132)：组合 Pi 公开的 edit/write 定义，保留原有工具接口并增加可选 `then_run`。执行时剥离该参数，把文件修改交回对应 `ctx.cwd` 的内置工具，再由公共助手执行后续命令。模型须在修改前决定命令，中间不再插入一次模型决策。
2. [`executeMutationThenRun` L95–125](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/then-run.ts#L95-L125)：先修改；没有命令就原样返回。有命令时先查文件，再调用 Pi bash，成功则追加 `succeeded` 标记与输出。修改抛错时命令 `skipped`；命令抛错时报告 `failed`，已写文件保留。它没有回滚事务语义。
3. [`resolveToolPath` 与 `withFusedFileQueue` L17–74](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/file-queue.ts#L17-L74)：统一相对路径、`@`、`~`、file URL，再以真实路径或现存祖先建立队列键。同一文件的队列覆盖修改和命令，`finally` 释放。外部写入不受此锁约束；两次 SHA-256 读取之间让出事件循环，仅能发现该窗口内可见的变化，不能证明命令全程独占文件。[检查窗口 L50–67](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/then-run.ts#L50-L67)。

命令字符串、取消信号和可选秒数超时交给 Pi bash；这里不设置默认超时。错误还须按发生阶段区分：非法编码的 file URL 在修改前就可能抛错，尚未进入生成 `skipped` 标记的分支。测试以调用计数确认这种路径既未修改文件也未运行命令。[参数 L16–30](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/then-run.ts#L16-L30)、[非法路径测试 L141–152](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/action-fusion-paths.test.ts#L141-L152)。

### ObservationPack：沿三个入口追踪

1. [`context` 回调 L137–209](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/index.ts#L137-L209)：复制消息数组，仅替换给提供方的投影。先存档，再计数；前两次完整发送，之后使用稳定占位符。计数按会话根目录与观察 ID 隔离；重启缺少内存计数时，以后续 assistant 消息数估算。存档或账本写入失败则保留原消息，但会话根目录解析在这个捕获范围之外。
2. [`createObservation` 与 `ensureStored` L67–156](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/observation.ts#L67-L156)：只处理成功、非空、全为文本块且合并后严格大于 10 KiB 的结果；错误、混合内容及含 EPR 收据行的结果跳过。ID 结合工具名、调用 ID 和内容散列；文本块用换行连接后存为 UTF-8。已有对象须通过大小和散列检查。占位符保留元数据及首尾各至多 512 字节的完整行，中段省略；这是有损摘录，精确回取依赖独立存档。[阈值 L12–17](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/observation.ts#L12-L17)、[占位符 L178–196](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/observation.ts#L178-L196)。
3. [`obs_recall` L41–103](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/index.ts#L41-L103)：校验 ID，按字节偏移读取，返回 `next_offset` 与 `eof`；含头部的输出上限为 16 KiB、400 行。底层裁剪页尾以免切断 UTF-8 字符；从零开始并沿返回偏移拼接，才能复原存档文本。任意偏移可能落在字符内部；回取本身也不重验内容散列。[分页 L207–250](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/observation.ts#L207-L250)。

归档位于当前会话目录的 `sol-pi/<sessionId>/observation-pack/objects/`，不需要提供方保存完整文本。替换投影保留消息的其他字段；原会话记录也未就地改写。对象目录及对象文件有符号链接检查，复用损坏对象会阻止打包。回取时未知 ID 或缺失文件报错，超出文件末尾的偏移也报错；这些失败不会伪造完整结果。[会话目录 L9–16](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/runtime-paths.ts#L9-L16)。

### 练习与证据边界

- 在临时副本把后续命令设为失败，预测错误标记、文件内容与下一次同文件操作顺序，再对照[失败及队列测试 L249–342](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/action-fusion.test.ts#L249-L342)。
- 构造跨页中文日志，预测四次投影，再去掉每页头部、依次拼接正文并比较 UTF-8 字节；参考[投影测试 L110–128](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/observation-pack.test.ts#L110-L128)与[分页测试 L261–285](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/observation-pack.test.ts#L261-L285)。
- 增加恰好等于阈值与多一字节的样本，解释为何只有后者进入打包。重启扩展后增减后续 assistant 消息，观察计数回退；由此区分代码采用的上下文投影次数代理与提供方实际计费请求数。

以上以源码阅读解释执行行为；实际运行结果见实验 004，练习变体尚待执行，不含性能结论。尤其[融合写入测试 L303–325](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/observation-pack.test.ts#L303-L325)的样本不足 10 KiB，只能证明直接传递；可把输出加长并将标记移出首尾摘录，检验占位符是否仍含标记。

## 可验证摘录与有条件的上下文压缩

本节依据固定提交 `d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0` 的实现与测试源码；测试引用描述读到的断言；实际执行范围和 5 项失败见实验 004，不把作者的基准成绩当作这里复现的结果。

### 1. Reducer：先存原文，再核对摘录

Evidence-Preserving Reducer 在 `tool_result` 上处理诊断命令的长日志，包括 `bash` 和融合 `edit/write` 的 `then_run` 输出。默认门槛为 4,096 字节，上限为 600,000 个 JavaScript 字符；命令不匹配、太短、太长或命中疑似秘密正则时跳过。完整输出文件只有满足系统临时目录、`pi-bash-*.log`、普通文件且非符号链接条件才会读取，否则使用内联内容；因此“原文”指实际取得的文本，不保证总能取得完整进程输出。[候选解析](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/candidate.ts#L34-L98) [筛选](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/index.ts#L64-L77) [默认值](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/config.ts#L14-L29)

合格文本先按 SHA-256 写入会话目录下 `objects/<前两位>/<hash>.txt`，再调用配置的 reducer 模型。回执包括源文件路径、源哈希、字节数、引用位置及引用哈希；主模型仍负责诊断、修复、重跑和判定，必要时按明确行或字节范围回读。融合调用只替换命令输出部分，保留修改确认和原有 `isError`。[归档](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/archive.ts#L24-L52) [回执格式](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts#L146-L176) [融合投影](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/candidate.ts#L34-L98)

### 2. “证据保留”验证了什么

验证器检查 JSON、schema、源哈希、与工具 `isError` 一致的状态、布尔型 `uncertain`、最多 12 条证据及每条最多 600 个 JavaScript 字符；每条引用必须是原文中的连续子串。它去重并定位第一次出现的行。失败日志若命中失败关键词正则，至少要有一条被标为 `fatal` 或 `failure` 的证据。[验证器](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts#L82-L143) [限额与正则](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/config.ts#L14-L29)

**这些检查证明摘录有出处，不证明语义完整。** 验证器不会确认 `kind` 标签是否准确、是否遗漏其他失败、是否抓住真正原因；失败 guard 也未检查该引用本身包含失败关键词。`uncertain=true` 仍可接受；“lossless”是提示语里的目标，不能据此宣称摘要无损。例如原文有两个失败，回执只准确引用其中一个，仍可能通过。这个例子是代码推演，非已执行测试。[验证器](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts#L82-L143)

模型不可用、调用异常、错误响应、无效回执、回执不比原文短时，扩展返回 `undefined`，保留原工具结果；无持久会话目录也跳过。父级取消信号会转交模型调用，另设 90 秒超时。**fail-open 有边界**：`archiveBody()` 在模型调用的 `try/catch` 之前，归档写入失败或同名文件内容不一致会抛出；本扩展没有把所有存储异常都转换成回退。[回退流程](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/index.ts#L77-L195) [信号传播](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/provider.ts#L74-L92) [归档完整性](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/archive.ts#L24-L52)

### 3. Compact：在完成步骤的边界停下，再续跑

Online Context Compact 的入口是一次把步骤新标为 `completed` 的 `update_plan`。`turn_end` 必须找到对应的成功工具结果，且助手没有 error/aborted、上下文信号未取消。通过经济或窗口条件后，还要用 Pi 的 `findCutPoint` 检查确有历史可压缩；随后调用 `abort()`，等整个 agent 在 `agent_settled` 空闲后才调用原生 `compact()`。因此窗口压力也不会绕过这条边界链路。[边界与选择](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L202-L311) [原生可行性](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L99-L148)

成功后发一个隐藏的 `triggerTurn: true` 提醒，要求为剩余工作重建计划；屏障等这个续跑也 settled，避免 print/JSON 模式提早结束。取消压缩或 `AbortError` 不触发续跑；其他压缩错误抛出，续跑没有启动也报错。压缩中禁止切换会话树。会话启动/树切换恢复当前分支最新有效状态并清除待执行动作；`steer` 或 `CORRECTION:` 清除旧计划、样本和债务。以上是恢复与取消保护，不是任意崩溃后都自动继续的保证。[续跑与错误](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L314-L440) [恢复与更正](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L178-L258) [状态持久化](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/state.ts#L133-L207)

### 4. 经济公式：缓存成本的启发式

令 `W=writeTokens`（压缩前上下文估计），`A=archiveTokens`，`M=memoTokens`，`r=cacheWriteReadRatio`，`D=carriedDebtTokens`：

```text
S = A − M
c = max(0, r − 1)
B = Wc / S                 # 本次盈亏平衡请求数，要求 S > 0
Bcombined = (D + Wc) / S   # 连同已有债务
```

运行时 `A=max(0,W−systemPromptEstimate−keepRecentTokens)`；保留尾部默认 20,000 token，备忘估计为 1,000 token。W 取可见消息加系统提示估计与有效 provider 报告的较大者。`r` 来自配置，发行入口默认 12.5，会话内不随模型切换重算；直接调用工厂而不传 r 则为 null，只能走窗口保护。[运行时估计](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L202-L311) [上下文估计](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L178-L258) [常量](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L35-L70) [比率配置](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L14-L35)

**从公式推断的假设**：`r−1` 将重建缓存的增量成本与原本可读的热缓存比较，以 cache-read token 等价量计账。代码没有观察实际命中率，也未将摘要模型的输入输出费用、额外延迟纳入上述 B；它是调度启发式，不是完整账单。冷缓存、命中率变化或计划预测失准，都可能使预估与实际节省不同。[成本公式](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/economics.ts#L14-L235) [配置说明](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/docs/configuration.md#L58-L65)

历史完成边界间请求数的均值为 μ，剩余边界数为 R。默认 `k=0`，使用 μ；若启用非零 k，样本不足 3 个时均值减半，否则用 `max(0,μ−k×样本标准差)`。令调整后均值为 L：

```text
Hraw = 1 + floor(L × max(0,R) × scale)       # 默认 scale=1
U = max(0, floor((window − context) / 平均正增长))
H = min(Hraw,U)                            # 无有效 U 时用 Hraw
Hfirst = min(2H,U)                         # 无 U 时不设此上限
```

第一次要求 `B≤Hfirst` 且预测请求数为正；后续要求 `B≤H`、`1.5B≤H`、`Bcombined≤H`。接近 `window−16,384` 时窗口保护可跳过经济门槛，但 `S>0` 与前述边界/原生可行性仍必须满足。这些倍数是当前实现常数，不是经证明的全局最优值。压缩后记录 Wc 债务与 S 的每请求偿还量；每次 `before_provider_request` 减一次，至零停止，属于估计记账而非真实付款。[预测与门槛](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/economics.ts#L14-L235) [偿还状态](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/state.ts#L133-L207)

### 5. 按真实公式手算

沿用经济测试的量级，令 `W=80,000, A=60,000, M=1,000, r=12.5, D=0`；已完成边界请求数 `[4,6,5]`，还有 4 个边界，窗口 200,000，平均每次正增长 2,000。于是 `S=59,000`，`Wc=920,000`，`B≈15.59`，`Hraw=21`，`U=60`，`H=21`。

| 情境 | 源码规则的结果 |
| --- | --- |
| 首次压缩 | `Hfirst=42`，15.59≤42，经济门槛通过。 |
| 后续压缩，其他输入相同 | `1.5B≈23.39>21`，返回 `deferred_subsequent_margin`。 |
| 后续压缩，剩余边界改为 6、已有债务 2,000,000 | H=31，margin 通过，但 `Bcombined≈49.49>31`，返回 `deferred_carried_debt`。 |

这是玩具算例的判定结果，不是节省金额或速度的测量。实际触发还取决于边界、窗口与原生可行性。[算例基底](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/online-context-compact-economics.test.ts#L12-L87) [实际公式](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/economics.ts#L14-L235)

### 6. 测试证据的范围

Reducer 测试断言准确引用、归档回读、融合输出保留修改确认，以及编造引用、模型错误、缺少模型/会话目录等回退；它们没有证明任意日志的语义完整性。Compact 经济测试覆盖请求预测、非正收益、无比率、窗口保护和已有债务。真实 `AgentSession` 测试检查一次/两次压缩后，原始 `prompt()` 等到最终续跑回复才返回；但使用 faux provider 和确定性摘要，因此验证的是生命周期，不是实际模型摘要质量、真实缓存命中或基准任务成绩。[Reducer 测试](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/evidence-preserving-reducer.test.ts#L239-L542) [经济测试](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/online-context-compact-economics.test.ts#L12-L87) [AgentSession 测试](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/online-context-compact-agent-session.test.ts#L47-L175)

## 与已有知识树的连接

- **Pi → SoL-Pi 是直接实现关系**：公开工具工厂、扩展事件、会话存储与原生 compaction 提供落点。先读基础循环，再追四个注册入口，避免把扩展行为误认成底座默认行为。
- **M09 推理 → M12 Harness**：前缀变化可能改变 prompt-cache 的复用与成本。这里的 cache write/read 比率是提供方计价启发式，与 [SGLang](sglang.md) 管理 GPU KV cache 的内存、调度机制有关联，但不是同一个指标或已实现的集成。
- **M12 → M14 评测**：与 [DeepSeek Harness](deepseek-harness.md) 的请求边界、[Harbor](harbor.md) 的任务与评分对照，研究同一任务的成功率、证据覆盖与总成本。SoL-Pi 本身没有为这些项目提供已验证的接线。
- **M12 → M13 Agent RL**：与 [APEX recipe](apex-agents-skyrl-recipe.md) 对照时，需要明确动作融合、结果投影、摘要如何改变轨迹和 reward 可见信息。这是后续实验方向，不是 SoL-Pi 已实现 RL 更新。
- **Claude Code / Agent SDK → SoL-Pi**：前者用于比较工具许可、进程 transport 与事件生命周期，后者展示可修改的效率机制。参见 [Claude 笔记](claude-agent-sdk.md)；不推定二者内部实现相同。

后续优先验证两项尚无证据的假设：缩短日志后是否仍能定位所有任务相关错误；把真实缓存未命中与摘要费用加入后，经济门槛是否仍能预测总成本下降。先建立质量判定与原文回取检查，再扩大任务规模。
