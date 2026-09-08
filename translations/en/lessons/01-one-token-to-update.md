<a id="首课从一个-token-的概率到一次正确的参数更新"></a>

# First lesson: from one token's probability to one correct parameter update

[Back to the learning roadmap](../ROADMAP.md) · [M01/M02 course](../curriculum/01-foundations-to-pretraining.md) · [Experiment script and results](../experiments/002-token-weighted-loss/README.md)

For readers with basic Python who want to understand what training actually changes. Allow an estimated 60–120 minutes for reading, hand calculations, and execution; this is a study-planning estimate. No GPU, PyTorch, model weights, or API is required.

This lesson is a worked example for M01/M02 and introduces a question about distributed normalization for M05. It uses three-dimensional logits as updatable parameters, omitting the Transformer that produces logits; it therefore does not implement attention, automatic differentiation, or a complete language model.

<a id="1-一次预测到底输出什么"></a>

## 1. What exactly does a prediction output?

Suppose the vocabulary contains only A, B, and C, numbered 0, 1, and 2. For a given context, the model outputs three unnormalized scores, called logits:

```text
z = [2, 1, 0]
Correct next token = B, so y = 1
```

Softmax converts scores into probabilities: `p_j = exp(z_j) / Σ exp(z_k)`. To avoid overflow from large scores, implementations first subtract the largest logit, then exponentiate; subtracting the same constant from every score leaves the probabilities unchanged.

| Token | Logit | Approximate probability |
|---|---:|---:|
| A | 2 | 0.665241 |
| B | 1 | 0.244728 |
| C | 0 | 0.090031 |

The model currently favors A, but the training label is B. Training here does not require randomly sampling an answer first; the supervised objective directly uses the probability of the correct label.

<a id="2-loss-给这个预测怎样打分"></a>

## 2. How does loss score this prediction?

Cross-entropy for a single target is `L = −log p_y`, using the natural logarithm. Here, `L ≈ 1.407606`. When the correct token's probability approaches 1, loss approaches 0; when the correct token is very unlikely, loss increases.

Calculate these yourself first: If the probabilities are `[1/3, 1/3, 1/3]`, what is the loss? If 1000 is added to every logit, should the probabilities and loss change? Do not look at the run results yet.

This is still only one position. A real decoder usually predicts the next token at many positions and also requires shifted labels, causal attention, and a mask for valid targets. M01 explains where these positions' logits come from.

<a id="3-梯度告诉我们怎样改变参数"></a>

## 3. Gradients tell us how to change parameters

The derivative with respect to the logits is `∂L/∂z_j = p_j − 1[j=y]`. The indicator function is 1 when j equals the correct label and 0 otherwise. The gradient here is approximately:

```text
[ 0.665241, -0.755272, 0.090031 ]
```

Why is B's component negative? Gradient descent applies `z_new = z − η × gradient`, so a negative gradient increases B's score while the other two scores decrease. With `η=0.1`, one step in this example lowers loss from **1.407606 to 1.307354**.

We update the logits directly here to make the direction clear. In a real neural network, logits are network outputs, and backpropagation must apply the chain rule to propagate this gradient to each layer's weights. A decrease in one step does not establish generalization, nor does it prove that every learning rate decreases the loss.

How can we check the derivative independently? Add and subtract a small h for each parameter and approximate the derivative with `(L(z+h)-L(z-h))/(2h)`. The experiment uses h=`1e-5` to compare finite differences with a manually implemented analytical gradient; it does not call autograd.

<a id="4-mask-决定哪些目标参与学习"></a>

## 4. Masks determine which targets participate in learning

Now construct six positions. To make the mathematics easy to check, all positions share the same three-dimensional parameter z; a real model's outputs usually differ across positions.

| Position | Correct label | Loss mask | Approximate per-position loss |
|---:|---|---:|---:|
| 0 | A | 1 | 0.407606 |
| 1 | C | 1 | 2.407606 |
| 2 | C | 1 | 2.407606 |
| 3 | C | 1 | 2.407606 |
| 4 | B | 0 | Excluded from the objective |
| 5 | A | 0 | Excluded from the objective |

There are 4 valid targets, so the overall loss should be:

`(0.407606 + 2.407606 × 3) / 4 ≈ 1.907606`

Do not divide by the six input positions. The experiment also checks that changing the final two masked labels leaves loss and gradients unchanged. If every target is masked, the script explicitly raises an error, requiring the caller to decide whether to skip or diagnose the batch, rather than dividing by zero or silently counting a successful update.

**A loss mask is not an attention mask.** In a real Transformer, a position whose own prediction loss is excluded can still provide context for later predictions, affecting other positions and network gradients. Tool observations are usually not supervised targets for agent actions, but can still provide context. Do not extrapolate “ignore the label” here into “the input token has no effect at all.”

<a id="5-设备更多时一个很隐蔽的错误"></a>

## 5. A subtle error when there are more devices

Put position 0 in group A and the remaining positions in group B. Group A has 1 valid target and group B has 3. Their mean losses are 0.407606 and 2.407606, respectively.

Averaging these two local means gives **1.407606**, which differs from the correct **1.907606**. The error comes from giving the one token in group A the same total weight as the three tokens in group B.

The correct method retains each group's valid count:

`L = (n_A × L_A + n_B × L_B) / (n_A + n_B)`

The gradient also changes: the correct shared-parameter gradient is approximately `[0.415241, 0.244728, -0.659969]`, while the incorrect averaging gives `[0.165241, 0.244728, -0.409969]`. This is therefore more than a small discrepancy on a dashboard: the model is optimizing a different weighted objective.

If every group has the same valid count, the incorrect method may happen to be correct. This explains why testing only equal-length data can miss the problem. The experiment changes the grouping, checks that correctly weighted loss/gradients remain unchanged, and includes a group in which all targets are masked.

“Incorrect” here is relative to the prespecified global token-average objective. If a study intentionally gives each group equal weight, that is a different objective, which must be stated explicitly and evaluated accordingly.

These “groups” are only lists in the same CPU process. With actual DDP/FSDP, you must also check whether the collective sums or averages and at which stage gradients are scaled; copying this formula directly can result in dividing by world size twice. See the [distributed training chapter](../handbook/02-distributed-pretraining-operations.md).

<a id="6-运行并读结果"></a>

## 6. Run the script and read the results

Run from the handbook repository root:

```powershell
python experiments/002-token-weighted-loss/run.py
```

The script prints and saves [results.json](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/002-token-weighted-loss/results.json). It has been executed in Python 3.11.4 / CPU in this round: maximum finite-difference errors for both a single position and the whole batch are below `2e-11`; three correct groupings agree; the incorrect-averaging counterexample, mask checks, and all-masked checks pass. The result file contains the date, full numerical values, and script hash.

Do not stop at “passed.” First locate the three functions `ce_and_grad`, `aggregate`, and `grouped` and explain each one's inputs and outputs; identify the line that determines effective-token weights and explain why finite differences help reveal derivation or coding errors.

<a id="7-读者练习与验收"></a>

## 7. Reader exercises and completion assessment

The following exercises are for learners; providing a reference script does not count as completing them:

1. Change the correct label to A, predict the gradient signs, then calculate. The answer should be approximately `[-0.334759, 0.244728, 0.090031]`.
2. Divide the four valid targets evenly between two groups and explain why the mean of means agrees in this case; then construct another unequal-length counterexample.
3. Compare “changing a masked label” with “changing a real model's context,” and explain why no effect in the former does not prove no effect in the latter.
4. Write caller-side rules for an all-masked batch: what to record, whether to advance the data position, and how to count successful updates; explain why these rules cannot substitute for the framework's actual behavior.

The uniform-probability answer is `log(3) ≈ 1.098612`; adding the same constant to all logits leaves the result unchanged. Complete the assessment by independently explaining these answers, checking one gradient and one grouping counterexample, and writing down real systems questions this experiment does not cover.

<a id="8-回到真实训练代码"></a>

## 8. Return to real training code

The formulas and numerical examples in this lesson are our own instructional construction. Continue into production interfaces through [OLMo label construction](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/data/utils.py#L590-L605) and [TorchTitan train_step](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/trainer.py#L860-L1003); for now, locate data, masks, and effective-token counts without needing to understand every parallel branch.

After the first lesson, complete the decoder in [M01](../curriculum/01-foundations-to-pretraining.md#m01), then study optimization in greater depth in [M02](../curriculum/01-foundations-to-pretraining.md#m02). When you later return to device groups in M05 and RL token loss in M11, you will encounter the same normalization question again.
