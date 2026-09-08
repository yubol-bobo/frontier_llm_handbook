<a id="实验-002一个-token-的梯度与全局有效-token-平均"></a>

# Experiment 002: one token's gradient and the global effective-token average

Accompanies the [first lesson](../../lessons/01-one-token-to-update.md), in course modules M01/M02/M05. Uses our own synthetic data and the Python standard library, with no upstream training-code imports and no need for PyTorch, networking, APIs, GPUs, or model weights.

Run from the repository root:

```powershell
python experiments/002-token-weighted-loss/run.py
```

The script validates numerical results before writing `results.json` in the same directory; rerunning updates this small result file of our own. Checks do not rely on Python `assert` and cannot be disabled by `-O`. Results record the execution date, Python version, and script-content hash.

<a id="验证问题"></a>

## Validation questions

1. Is stable cross-entropy unchanged when the same constant is added to all logits? Does the manually implemented gradient agree with finite differences?
2. Does a selected small gradient step reduce this toy objective?
3. With four valid targets among six positions, do loss/gradients weighted by valid counts agree under different groupings?
4. Does deliberately incorrect averaging of local means change the objective and gradient?
5. Does changing masked target labels have no effect? Is an all-masked case handled explicitly?

This toy treats all positions' logits as a shared three-dimensional parameter; it does not imply that different positions in a real Transformer have identical logits. Groups are computed in the same CPU process, validating a mathematical reduction rather than DDP/NCCL, actual attention, automatic differentiation, or large-model training. The incorrect algorithm is retained as a counterexample, not a recommended implementation.

The [results.json](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/002-token-weighted-loss/results.json) in this directory stores actual results; other exercises in the curriculum still require their own execution records.
