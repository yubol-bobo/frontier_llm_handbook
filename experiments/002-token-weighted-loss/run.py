"""CPU teaching example: stable CE, finite differences and partition weighting.

Only the Python standard library is used. No model, autograd engine, distributed
runtime or upstream package is imported. All examples use synthetic class labels.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
from datetime import datetime, timezone


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ce_and_grad(logits, target):
    require(bool(logits) and 0 <= target < len(logits), "Invalid target or logits")
    require(all(math.isfinite(z) for z in logits), "Logits must be finite")
    peak = max(logits)
    exp_shifted = [math.exp(z - peak) for z in logits]
    normalizer = sum(exp_shifted)
    probs = [x / normalizer for x in exp_shifted]
    loss = (peak - logits[target]) + math.log(normalizer)
    grad = [p - float(i == target) for i, p in enumerate(probs)]
    return loss, grad, probs


def aggregate(theta, rows):
    """Every synthetic position shares theta; only mask=1 receives CE loss."""
    count = sum(mask for _, mask in rows)
    require(count > 0, "No effective targets: define skip/error before dividing")
    loss_sum = 0.0
    grad_sum = [0.0] * len(theta)
    for target, mask in rows:
        require(mask in (0, 1), "Only binary masks are used in this lesson")
        if not mask:
            continue
        loss, grad, _ = ce_and_grad(theta, target)
        loss_sum += loss
        grad_sum = [a + b for a, b in zip(grad_sum, grad)]
    return loss_sum / count, [g / count for g in grad_sum], count


def grouped(theta, rows, partition, weighted):
    require(sorted(i for group in partition for i in group) == list(range(len(rows))),
            "Partition must contain each row exactly once")
    pieces = []
    for group in partition:
        subset = [rows[i] for i in group]
        if sum(mask for _, mask in subset):
            pieces.append(aggregate(theta, subset))
    require(bool(pieces), "No effective targets in any group")
    weights = [n if weighted else 1 for _, _, n in pieces]
    scale = sum(weights)
    loss = sum(w * item[0] for w, item in zip(weights, pieces)) / scale
    grad = [sum(w * item[1][j] for w, item in zip(weights, pieces)) / scale
            for j in range(len(theta))]
    return loss, grad


def finite_difference(fn, theta, h=1e-5):
    result = []
    for j in range(len(theta)):
        plus, minus = list(theta), list(theta)
        plus[j] += h
        minus[j] -= h
        result.append((fn(plus) - fn(minus)) / (2 * h))
    return result


def max_error(xs, ys):
    return max(abs(a - b) for a, b in zip(xs, ys))


def main():
    theta = [2.0, 1.0, 0.0]
    loss, grad, probs = ce_and_grad(theta, 1)
    numeric = finite_difference(lambda t: ce_and_grad(t, 1)[0], theta)
    require(max_error(grad, numeric) < 1e-8, "Single-position derivative mismatch")
    shifted_loss = ce_and_grad([1002.0, 1001.0, 1000.0], 1)[0]
    require(abs(loss - shifted_loss) < 1e-12, "Stable shift invariance failed")
    learning_rate = 0.1
    updated = [t - learning_rate * g for t, g in zip(theta, grad)]
    after = ce_and_grad(updated, 1)[0]
    require(after < loss, "Chosen toy gradient step did not reduce CE")

    # Group A has 1 supervised position; group B has 3 plus 2 ignored positions.
    rows = [(0, 1), (2, 1), (2, 1), (2, 1), (1, 0), (0, 0)]
    global_loss, global_grad, count = aggregate(theta, rows)
    global_numeric = finite_difference(lambda t: aggregate(t, rows)[0], theta)
    require(max_error(global_grad, global_numeric) < 1e-8, "Batch derivative mismatch")
    uneven = [[0], [1, 2, 3, 4, 5]]
    balanced = [[0, 1], [2, 3, 4, 5]]
    with_empty_group = [[0, 1, 2, 3], [4, 5]]
    reductions = {}
    for name, partition in [("uneven", uneven), ("balanced", balanced),
                            ("masked_only_group", with_empty_group)]:
        got_loss, got_grad = grouped(theta, rows, partition, weighted=True)
        require(abs(got_loss - global_loss) < 1e-12 and
                max_error(got_grad, global_grad) < 1e-12,
                "Weighted result changed with partition: " + name)
        reductions[name] = {"loss": got_loss, "gradient": got_grad}

    naive_loss, naive_grad = grouped(theta, rows, uneven, weighted=False)
    require(abs(naive_loss - global_loss) > 0.4 and
            max_error(naive_grad, global_grad) > 0.2,
            "Counterexample no longer distinguishes the wrong objective")
    changed_masked = rows[:4] + [(2, 0), (2, 0)]
    masked_loss, masked_grad, _ = aggregate(theta, changed_masked)
    require(masked_loss == global_loss and masked_grad == global_grad,
            "Ignored target labels changed the objective")
    try:
        aggregate(theta, [(0, 0), (1, 0)])
    except RuntimeError as exc:
        require("No effective targets" in str(exc), "Unexpected empty-batch error")
    else:
        raise RuntimeError("All-masked batch was silently accepted")

    report = {
        "schema_version": 1,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "stdlib CPU arithmetic; no autograd, decoder, GPU or distributed execution",
        "single_position": {"logits": theta, "target": 1, "probabilities": probs,
                            "loss": loss, "gradient": grad,
                            "finite_difference_max_error": max_error(grad, numeric),
                            "learning_rate": learning_rate, "loss_after_step": after},
        "batch": {"rows_target_mask": rows, "effective_targets": count,
                  "loss": global_loss, "gradient": global_grad,
                  "finite_difference_max_error": max_error(global_grad, global_numeric),
                  "correct_partition_results": reductions,
                  "wrong_mean_of_means_loss": naive_loss,
                  "wrong_mean_of_means_gradient": naive_grad},
        "checks_passed": ["stable logit shift", "single-position finite differences",
                          "toy gradient step reduces loss", "batch finite differences",
                          "three weighted partitions agree", "wrong reduction differs",
                          "ignored labels invariant", "all-masked batch explicitly rejected"],
    }
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
