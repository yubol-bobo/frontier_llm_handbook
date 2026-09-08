# Claude Code / Agent SDK：控制协议、权限与 harness 的实现边界

日期：2026-09-08。阶段：L1 局部源码审读。官方仓库：[Claude Agent SDK for Python](https://github.com/anthropics/claude-agent-sdk-python)，固定 SHA：`f1315c69a74db1c15fed2e5974918495d90b7d57`。

本地取得官方 SDK 的 144 个 tracked files；未安装 SDK、启动 Claude、调用模型 API 或运行上游测试。课程原创 CPU 实验的结果另见 [实验说明](../../experiments/harness-state-machine/README.md)。历史镜像的观察与来源边界另列在本文后半部分。

## 先确定读的是哪一层

Python SDK 通过 transport 与 Claude Code CLI 通信。SDK 中能直接读到配置转换、消息处理、权限回调、hooks 和进程生命周期代码；它不是完整 CLI 内部源码。官方发行包会捆绑 CLI，[README](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/README.md#L15-L21) 和 [CLI 定位实现](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L247-L263) 给出这一边界。源码浅克隆也不等于已经下载了发行包中的 CLI。

学习顺序：[Pi](pi.md) → [Claude Code 进阶讲解](../../handbook/05-claude-code-harness.md) → 本文调用链 → CPU 故障实验 → [APEX 轨迹与训练接口](apex-agents-skyrl-recipe.md)。后两个项目之间是教学上的概念连接，不是已实现的集成。

## 按一次控制请求读源码

1. **配置变成 CLI 参数。** 从 [`_build_command`](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L562-L586) 读 `stream-json` 和 system prompt 参数；再读 [resume/session 参数](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L632-L645) 与 [fork 参数](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L693-L711)。参数转发证明接口接线，不证明内部恢复算法或外部副作用恰好执行一次。
2. **CLI 发出权限请求。** [`_handle_control_request`](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/query.py#L469-L529) 按 subtype 分派；`can_use_tool` 分支建立含 tool/agent 信息的 context，调用用户回调，把 Allow/Deny 结果编码为控制响应。Allow 可以返回修改后的 input，Deny 可以携带 interrupt。这里的 `signal=None` 还表明不能从回调类型名推断已有完整取消信号支持。
3. **响应需要双向通道存活。** [输入流生命周期](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/query.py#L819-L861) 在 SDK MCP、hooks 或权限回调需要双向通信时等待结果边界后才结束输入。注释明确提醒一个 result 不必然是整个后台任务生命周期结束，并记录多消息输入的限制；本次没有执行该边界场景。

## 能推出什么，不能推出什么

| 固定源码事实 | 工程含义与验证边界 |
|---|---|
| 回调结果被编码为 allow/deny 控制响应 | 权限是执行协议的一部分；仍须分别验证 CLI 的所有执行路径、参数检查与操作系统 sandbox |
| resume 和 fork 被转换为 CLI 选项 | 会话身份属于运行契约；不能据此宣称自动恢复外部工具状态 |
| 控制请求依赖 stdin/stdout 生命周期 | 不能在第一个文本输出后随意关闭输入；本地真实进程取消与后台任务清理仍待测试 |
| SDK 与 CLI 是不同版本的产物 | 复现实验要记录 SDK commit/package、CLI、模型、设置和权限，不只保存一个 SDK SHA |

课程提出的故障问题是：工具已经改变环境，但结果尚未写入 journal，重启时是否应该重放？我们的 [状态机实验](../../experiments/harness-state-machine/README.md) 将其标成 unknown outcome 并停止自动重试。它是保守的教学设计；不是对 Claude Code 内部实现的断言，也不提供跨进程 exactly-once 保证。

## 与训练知识树连接

对照 [DeepSeek Harness](deepseek-harness.md) 的日志投影和 Pi 的工具事件，区分完整事件记录、下一次模型上下文和可训练 token 序列。进入 Agent RL 时要额外保存模型实际采样的 token IDs、logprobs、loss masks、策略版本及分支信息；SDK 的文本消息不能被直接当作这些量的替代品。

第一轮交付物：画出 Python SDK / CLI / 模型 / 工具四个边界，标注权限响应位置；给出一个“已开始但结果未知”的故障 trace；解释上下文压缩怎样改变下一步的输入。完成这些后再设计真实 SDK 对照实验，保持任务、权限与预算一致。

## 官方资料与阅读状态

- [Agent loop 官方文档](https://code.claude.com/docs/en/agent-sdk/agent-loop)：公开接口与当前行为说明，核查于 2026-09-08；不是固定历史实现。
- [上下文工程](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：compaction、笔记、检索和子任务的设计取舍。
- [长任务 harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)：初始化、进度产物与独立验收的作者实验。

## Claude Code 历史镜像：有限范围证据笔记

核查日期：2026-09-08。本材料用于静态阅读第三方历史镜像；它不是经过认证的 Anthropic 发布版本，也不能证明当前生产行为。本次没有安装或运行镜像代码及依赖，本文不包含源文件。

### 快照选择与来源边界

选用镜像提交 `f5a40b86dede580f6543bf8926c9af017eea9409`。GitHub 的[该提交历史](https://github.com/jaicorn/claude-code-source/commits/f5a40b86dede580f6543bf8926c9af017eea9409/)显示：2026 年 3 月 31 日有一条提交，归于 `realsigridjin`，标题为 “init: add source code from src.zip”。[根目录](https://github.com/jaicorn/claude-code-source/tree/f5a40b86dede580f6543bf8926c9af017eea9409)仅包含 `src/`。

README 宣称的 `backup` 路径不能作为本次可用快照：`jaicorn/claude-code-source/tree/backup` 与 README 实际指向的 `nirholas/claude-code/tree/backup` 均返回 404；检查到的分支列表只列出 `main`。因此不能把所选 SHA 称为 backup 分支。

SHA 固定了镜像内容及其仓库内历史。本次没有取得经独立认证的 Anthropic 原始文件、签名清单或校验和并进行比对，所以 SHA 与导入说明均不能证明逐字节一致、内容完整或构建时启用了哪些功能。仓库把材料标为 Claude Code v2.1.88；这仍是镜像方的版本归属说明。

### 三处小范围阅读入口

以下行号通过 GitHub blame 页的实际源文件行号核对，没有使用网页工具归一化后的文本行号。

| 入口 | 可直接观察的静态事实 | 可迁移的设计问题 |
| --- | --- | --- |
| [query.ts，第 826–862 行](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/query.ts#L826-L862) | 助手消息中的工具调用块设置继续处理标记；流式执行器接收工具块，并把已完成结果转换成 API 所需的消息形式。 | 宿主程序怎样把模型提出的动作与实际结果组织成下一次模型输入？ |
| [autoCompact.ts，第 28–49 行](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/compact/autoCompact.ts#L28-L49)、[62–91 行](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/compact/autoCompact.ts#L62-L91)、[257–265 行](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/compact/autoCompact.ts#L257-L265) | 有效上下文容量先预留输出空间；额外缓冲控制自动压缩时机；连续失败检查用于终止反复压缩尝试。 | 恢复动作本身需要预留多少容量？连续失败后何时停止？其中数字属于历史实现细节，不是通用推荐参数。 |
| [toolExecution.ts，第 916–931 行](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/tools/toolExecution.ts#L916-L931)、[995–999 行](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/tools/toolExecution.ts#L995-L999) | 宿主等待权限解析器返回决定，输入包括工具、处理后的参数、上下文和权限回调；随后区分非允许结果。 | 模型动作建议在什么位置转化为授权决定？这几行不能证明整个权限系统正确。 |

### 镜像维护者增补与通知记录

[初始 services 目录](https://github.com/jaicorn/claude-code-source/tree/f5a40b86dede580f6543bf8926c9af017eea9409/src/services)没有 `x402` 目录；[较晚的 d43bd40 目录](https://github.com/jaicorn/claude-code-source/tree/d43bd40690853fd323758e038cb686930d53a39f/src/services)包含该目录。这只是目录级对照，不是完整真实性审计。

在[固定于 e37f8a9 的维护者更正](https://github.com/nirholas/fresh-start/blob/e37f8a9ea1a7ab4d83432d93bd357a332549e924/HISTORY.md)中，nirholas 说明 x402 出自自己，并把生成文档、网页终端、兼容层、构建类型声明、Docker 与探索用 MCP 服务列为增补。他明确把核查范围限定到 2026-03-31T12:43Z。应把这类内容标为维护者第一人称来源说明，不要归属于 Anthropic。

[3 月 31 日通知](https://github.com/github/dmca/blob/e8211dc62ad9c3a23d3f5c2878a4557d6b9f16fd/2026/03/2026-03-31-anthropic.md)指控 nirholas 仓库侵权，并说明相关作品没有采用开源许可证；GitHub 说明整网处理影响了约 8,100 个仓库。[4 月 1 日部分撤回](https://github.com/github/dmca/blob/615484fed194981ef67a284ffe08c5625839e077/2026/04/2026-04-01-anthropic-retraction.md)保留了针对父仓库及原通知逐一列出的 96 个 fork URL 的通知，同时要求恢复其他被整网处理的仓库。部分撤回不等于源代码授权，也不构成真实性认证。

公开手册可保留原创分析、小范围固定链接与清晰的来源标签，不应把镜像、其生成文档或提示词导入为手册资产。这三处入口可支撑历史 harness 案例，不能支撑关于当前 Claude Code 实现、当前功能开关或专有模型内部机制的结论。
