<a id="harbor一次-agent-行为怎样成为可信的-reward"></a>

# Harbor: How an agent's behavior becomes a trustworthy reward

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: `https://github.com/harbor-framework/harbor` @ `9a2e3b135cc8fb1e41131020f83370cb2f12ae93`  
Verification scope: Actually read the local single-step trial, shared/separate verifier lifecycles, and reward-parsing implementation; did not run containers, agents, evaluations, or RL. Read upstream `AGENTS.md` and did not modify upstream code.

<a id="核心问题"></a>

## Core question

Environment execution failure, agent timeout, an incorrect answer, and a corrupted reward file are four different problems. How does Harbor preserve enough evidence and send them down the correct result paths? This is closer to environment engineering for agent RL than simply running a test at the end.

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

All local paths are relative to the learning repository root. The following links are pinned to the commit actually cloned.

| Evidence | Local implementation | Pinned source |
|---|---|---|
| H1 | `sources/harbor/src/harbor/trial/single_step.py` | [SingleStepTrial._run and _run_agent, lines 38–87](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/trial/single_step.py#L38-L87) |
| H2 | `sources/harbor/src/harbor/trial/trial.py` | [shared / separate verifier, lines 641–744](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/trial/trial.py#L641-L744) |
| H3 | `sources/harbor/src/harbor/verifier/verifier.py` | [Reward parsing and verify, lines 67–266](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/verifier/verifier.py#L67-L266) |
| H4 | `sources/harbor-cookbook/pyproject.toml` | [Cookbook declares a harbor dependency, line 11](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/pyproject.toml#L11) |

Main chain: `SingleStepTrial._run()` → `_run_agent()` → upload agent logs → collect artifacts → `_run_verifier()` → `VerifierFactory` → `Verifier.verify()` → `VerifierResult(rewards=...)`. Separate-verifier mode stops the agent environment first, then creates a verification environment and passes artifacts to it. Shared mode verifies in the same environment and stops the agent environment afterward.

`Verifier.verify()` first locates and uploads tests, merges verifier environment variables, and executes the test script through the environment interface; non-mounted environments also require downloading verification logs. It reads `reward.json` preferentially, falling back to `reward.txt`, and raises `RewardFileNotFoundError` if neither exists. The return value comes from the reward-file protocol, rather than directly treating the test process's exit code as a reward.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Source fact: verification can still occur after an agent timeout.** `_run_agent()` catches `AgentTimeoutError` and `NonZeroAgentExitCodeError`, records the exception, and synchronizes output in finally. The subsequent chain can still collect partial artifacts and execute the verifier. This short code segment does not establish that other uncaught exceptions receive the same treatment.
- **Source fact: environment boundaries are part of configuration.** The shared path reuses the agent environment; the separate path transfers artifacts and sets `skip_tests_upload=True`, with tests supplied by the verifier image. Both paths set the actual verifier user, phase-specific network policy, and timeout.
- **Source fact: reward content is validated on input.** Text parsing rejects empty files and non-finite numbers; JSON parsing checks numeric types and finiteness to keep NaN/Infinity out of downstream data. This check does not guarantee answer correctness, which is still defined by task tests.
- **Source fact: the default verifier preserves the reward dictionary.** The end of H3 directly constructs `VerifierResult(rewards=rewards)`; `models/verifier/result.py:4–5` is also just a dictionary field, with no automatic conversion of multiple dimensions into a scalar named `reward`. Training must explicitly specify which entry to read or how to aggregate them. The repository also has a Rewardkit aggregation mechanism, which must not be mistaken for unconditional behavior of the default verifier.
- **Reading inference:** A separate verification environment clarifies the boundary between the evaluated execution environment and the evaluation environment, at the cost of images, artifact protocols, and transfer overhead. Omissions from the artifact list change what can be verified. The presence of separate mode alone does not establish that all tasks resist reward hacking.

<a id="与其他项目的连接"></a>

## Connections to other projects

- **Direct dependency, in the direction Cookbook → Harbor:** H4 confirms the dependency declaration; Harbor does not depend on Cookbook. The [Cookbook study note](harbor-cookbook.md) can serve as an entry point for task creation.
- **Example integration has version boundaries:** Cookbook's `harbor_cookbook/harbor_rl/train.py:3` separately specifies the `feature/harbor-rl-4d0` branch. The Harbor main snapshot in this study lacks the `harbor.rl` referenced by that example; consequently, main's verifier source cannot establish the complete behavior of `RLEnvironment.grade()` on that branch.
- **Conceptual correspondence:** [Verifiers](verifiers.md) also turns environment interactions into feedback. This reading did not review its complete bridge to Harbor, so conceptual correspondence is not drawn as a direct dependency.
- **Conceptual correspondence:** [SGLang](sglang.md) provides generation execution and throughput mechanisms; Harbor handles tasks/environments/verification in the chain traced here. Being usable in the same training system does not establish out-of-the-box compatibility between arbitrary versions.

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** Create a small task that only modifies text artifacts, with fixed instructions, image contents, and verifier. Run a normal agent, an agent that writes partial artifacts before timing out, and a test script that generates an invalid reward; then compare shared / separate modes.

Inputs: identical initial files, artifact paths, and verification rules. Controls: resource limits, agent budget, random seed/model sampling settings, and task version. Observe: agent exception, artifact completeness, verifier exception, reward fields, and phase durations. Expected: partial artifacts can be graded; invalid rewards are distinguished as parsing errors rather than silently becoming zero scores. Actual result: not executed; no success-rate data. Compute: Docker and an ordinary CPU suffice for a scripted agent; an LLM agent additionally requires a model API or local inference.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Continue into `trial/multi_step.py` to confirm intermediate rewards, early stopping, and final-result aggregation.
- [ ] Read `trial/artifact_handler.py` to trace missing artifacts, filtering rules, and transfer-failure semantics.
- [ ] Start with a minimal Cookbook recipe, verify the task protocol first, then connect an RL framework.
