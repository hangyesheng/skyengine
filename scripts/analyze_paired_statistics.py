#!/usr/bin/env python3
"""Compute descriptive paired statistics for formal clean/uncertain results.

The input is one or more paired CSV files produced by
``summarize_formal_results.py``.  The script reports bootstrap confidence
intervals for the mean degradation and an exact two-sided sign-test p-value.
It also compares each method with a selected reference on the common
``(family, instance, run, seed)`` keys. These tests are deliberately small-sample
descriptive evidence; they do not replace a larger repeated experiment.
"""

import argparse
import csv
import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path


def read_method(spec):
    if "=" not in spec:
        raise ValueError(f"方法参数必须是 LABEL=CSV_PATH: {spec}")
    label, path_text = spec.split("=", 1)
    path = Path(path_text)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    seen = set()
    for row in rows:
        try:
            clean = float(row["clean_makespan"])
            uncertain = float(row["uncertain_makespan"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: 无法解析 makespan") from exc
        if not math.isfinite(clean) or clean <= 0 or not math.isfinite(uncertain):
            raise ValueError(f"{path}: 存在无效 makespan")
        item = dict(row)
        item.update({
            "method": label,
            "clean": clean,
            "uncertain": uncertain,
            "delta": uncertain - clean,
            "relative": (uncertain - clean) / clean,
        })
        if row.get("seed") in (None, ""):
            raise ValueError(f"{path}: 缺少测试 seed，无法进行跨方法配对")
        item["key"] = (row.get("family", ""), row.get("instance", ""),
                       row.get("run", ""), row["seed"])
        if item["key"] in seen:
            raise ValueError(f"{path}: 重复的测试配对键 {item['key']}")
        seen.add(item["key"])
        parsed.append(item)
    return label, parsed, path


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / len(values))


def percentile(values, probability):
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(values, seed, repetitions=10000):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    samples = []
    n = len(values)
    for _ in range(repetitions):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(samples, 0.025), percentile(samples, 0.975)


def exact_sign_pvalue(values):
    """Two-sided exact sign test against a zero median, ignoring ties."""
    signs = [1 if value > 0 else -1 for value in values if value != 0]
    n = len(signs)
    if n == 0:
        return 1.0
    positives = sum(sign > 0 for sign in signs)
    tail = sum(math.comb(n, k) for k in range(positives + 1)) / (2 ** n)
    upper = sum(math.comb(n, k) for k in range(positives, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * min(tail, upper))


def fmt(value):
    return "" if value is None or not math.isfinite(value) else f"{value:.6f}"


def write_within(method_rows, output, bootstrap_repetitions):
    fields = [
        "method", "family", "n_pairs", "n_instances", "n_seeds",
        "clean_mean", "clean_std", "uncertain_mean", "uncertain_std",
        "delta_mean", "delta_std", "delta_ci95_low", "delta_ci95_high",
        "relative_mean", "relative_std", "relative_ci95_low", "relative_ci95_high",
        "positive_degradation_rate", "delta_sign_test_p",
    ]
    grouped = defaultdict(list)
    for label, rows in method_rows.items():
        for row in rows:
            grouped[(label, row.get("family", ""))].append(row)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, ((label, family), rows) in enumerate(sorted(grouped.items())):
            clean = [row["clean"] for row in rows]
            uncertain = [row["uncertain"] for row in rows]
            delta = [row["delta"] for row in rows]
            relative = [row["relative"] for row in rows]
            delta_low, delta_high = bootstrap_mean_ci(delta, 1000 + index, bootstrap_repetitions)
            relative_low, relative_high = bootstrap_mean_ci(relative, 2000 + index, bootstrap_repetitions)
            writer.writerow({
                "method": label,
                "family": family,
                "n_pairs": len(rows),
                "n_instances": len({row.get("instance") for row in rows}),
                "n_seeds": len({row["seed"] for row in rows}),
                "clean_mean": fmt(mean(clean)),
                "clean_std": fmt(std(clean)),
                "uncertain_mean": fmt(mean(uncertain)),
                "uncertain_std": fmt(std(uncertain)),
                "delta_mean": fmt(mean(delta)),
                "delta_std": fmt(std(delta)),
                "delta_ci95_low": fmt(delta_low),
                "delta_ci95_high": fmt(delta_high),
                "relative_mean": fmt(mean(relative)),
                "relative_std": fmt(std(relative)),
                "relative_ci95_low": fmt(relative_low),
                "relative_ci95_high": fmt(relative_high),
                "positive_degradation_rate": fmt(sum(value > 0 for value in delta) / len(delta)),
                "delta_sign_test_p": fmt(exact_sign_pvalue(delta)),
            })


def write_comparisons(method_rows, reference, output, bootstrap_repetitions):
    if reference not in method_rows:
        raise ValueError(f"参考方法不存在: {reference}")
    fields = [
        "reference_method", "method", "family", "n_pairs", "n_instances", "n_seeds",
        "clean_diff_mean", "clean_diff_ci95_low", "clean_diff_ci95_high", "clean_diff_sign_test_p",
        "uncertain_diff_mean", "uncertain_diff_ci95_low", "uncertain_diff_ci95_high", "uncertain_diff_sign_test_p",
        "delta_diff_mean", "delta_diff_ci95_low", "delta_diff_ci95_high", "delta_diff_sign_test_p",
    ]
    ref_by_key = {row["key"]: row for row in method_rows[reference]}
    grouped = defaultdict(list)
    for label, rows in method_rows.items():
        if label == reference:
            continue
        for row in rows:
            base = ref_by_key.get(row["key"])
            if base is None:
                continue
            grouped[(label, row.get("family", ""))].append((row, base))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, ((label, family), pairs) in enumerate(sorted(grouped.items())):
            clean = [row["clean"] - base["clean"] for row, base in pairs]
            uncertain = [row["uncertain"] - base["uncertain"] for row, base in pairs]
            delta = [row["delta"] - base["delta"] for row, base in pairs]
            values = {}
            for prefix, sequence, offset in (("clean", clean, 3000), ("uncertain", uncertain, 4000), ("delta", delta, 5000)):
                low, high = bootstrap_mean_ci(sequence, offset + index, bootstrap_repetitions)
                values[f"{prefix}_diff_mean"] = fmt(mean(sequence))
                values[f"{prefix}_diff_ci95_low"] = fmt(low)
                values[f"{prefix}_diff_ci95_high"] = fmt(high)
                values[f"{prefix}_diff_sign_test_p"] = fmt(exact_sign_pvalue(sequence))
            writer.writerow({"reference_method": reference, "method": label, "family": family,
                             "n_pairs": len(pairs),
                             "n_instances": len({row.get("instance") for row, _ in pairs}),
                             "n_seeds": len({row["seed"] for row, _ in pairs}), **values})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True,
                        help="方法与配对 CSV，格式 LABEL=path.csv；可重复传入")
    parser.add_argument("--reference-method", required=True, help="方法比较的参考方法标签")
    parser.add_argument("--within-output", required=True, help="按方法/数据族的配对统计 CSV")
    parser.add_argument("--comparison-output", required=True, help="相对参考方法的配对差异 CSV")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        parser.error("--bootstrap-repetitions 必须为正数")
    method_rows = {}
    source_hashes = {}
    for spec in args.method:
        label, rows, path = read_method(spec)
        if label in method_rows:
            raise ValueError(f"重复方法标签: {label}")
        method_rows[label] = rows
        source_hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    within = Path(args.within_output)
    comparison = Path(args.comparison_output)
    within.parent.mkdir(parents=True, exist_ok=True)
    comparison.parent.mkdir(parents=True, exist_ok=True)
    write_within(method_rows, within, args.bootstrap_repetitions)
    write_comparisons(method_rows, args.reference_method, comparison, args.bootstrap_repetitions)
    print(f"within={within}")
    print(f"comparison={comparison}")
    print(f"reference={args.reference_method}")
    print(f"source_sha256={source_hashes}")


if __name__ == "__main__":
    main()
