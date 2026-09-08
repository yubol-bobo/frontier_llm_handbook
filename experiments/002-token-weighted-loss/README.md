# 实验 002：一个 token 的梯度与全局有效 token 平均

对应 [首课讲解](../../lessons/01-one-token-to-update.md)，课程 M01/M02/M05。使用自有合成数据和 Python 标准库，不导入上游训练代码，不需 PyTorch、网络、API、GPU 或模型权重。

在仓库根目录执行：

```powershell
python experiments/002-token-weighted-loss/run.py
```

脚本先验证数值再写入同目录 `results.json`，重跑会更新这份自有小结果。检查不依赖 Python `assert`，不会因 `-O` 被关闭。结果记录执行日期、Python 版本和脚本内容 hash。

## 验证问题

1. 稳定交叉熵在 logits 同加常数后是否不变？手写梯度是否符合有限差分？
2. 一个选定的小梯度步是否降低此 toy objective？
3. 六个位置中四个有效目标，按不同方式分组后，按有效数量加权的 loss/梯度是否一致？
4. 故意错误的局部平均再平均，是否改变 objective 和梯度？
5. 修改被 mask 的目标标签是否无影响？全 mask 是否得到明确处理？

这个 toy 把所有位置的 logits 视为共享三维参数，并不表示真实 Transformer 的不同位置 logits 相同。分组在同一 CPU 进程中计算，验证的是数学归约，不是 DDP/NCCL、真实 attention、自动微分或大模型训练。错误算法作为反例保留，不用于建议实现。

本目录的 [results.json](results.json) 保存实际结果；课程中其他练习仍需各自执行记录。
