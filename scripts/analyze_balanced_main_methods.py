#!/usr/bin/env python3
"""Analyze the balanced three-method main comparison.

The raw learning-method results contain three training checkpoints evaluated
under the same three test seeds.  For method-to-method paired comparisons we
first average those checkpoints at each (family, instance, test seed), then
compare the resulting common test conditions.  This avoids treating the same
OR-Tools run as three independent observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
from collections import defaultdict
from pathlib import Path


def derive_trial_seed(base_seed: int, family: str, instance: str, run: int) -> int:
    identity = f"{base_seed}:{family}:{instance}:{run}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def identify_test_seed(row: dict) -> int:
    actual = int(row["seed"])
    candidates = [seed for seed in (601, 602, 603)
                  if derive_trial_seed(seed, row["family"], row["instance"], int(row["run"])) == actual]
    if len(candidates) != 1:
        raise ValueError(f"cannot identify test seed for {row}: {candidates}")
    return candidates[0]


def read_rows(spec: str) -> tuple[str, list[dict]]:
    label, path_text = spec.split("=", 1)
    rows = []
    with Path(path_text).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["method"] = label
            row["clean"] = float(row["clean_makespan"])
            row["event"] = float(row["uncertain_makespan"])
            row["delta"] = row["event"] - row["clean"]
            row["relative"] = row["delta"] / row["clean"]
            row["test_seed"] = identify_test_seed(row)
            rows.append(row)
    return label, rows


def training_seed(row: dict) -> str:
    method = row["method"]
    if method == "OR-Tools":
        return "none"
    source = row.get("clean_experiment_id", "")
    match = re.search(r"(?:seed|m)(10[1-3])", source)
    if match:
        return match.group(1)
    # Fallback for source IDs that do not encode the checkpoint seed.
    known = {
        "196b717a5ac8dfa13bcac803b9c4d238d810578b43498c03ad3cde2ed49ff33c": "101",
        "a219cd02d9a442e147b9b409aaa7df5e2ca64509a62ff36e6b9da340a34e51c0": "102",
        "1492789d9e31990c4ae8e8d18624f198b50ce2475ce08c7fddd7bfc431e32cba": "103",
    }
    if row.get("model_sha256") in known:
        return known[row["model_sha256"]]
    raise ValueError(f"cannot identify training seed: {row}")


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def percentile(values, probability):
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * probability
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (pos - low)


def bootstrap(values, seed, repetitions=10000):
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    samples = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(repetitions)]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def sign_pvalue(values):
    values = [value for value in values if value != 0]
    n = len(values)
    if not n:
        return 1.0
    positive = sum(value > 0 for value in values)
    lower = sum(math.comb(n, k) for k in range(positive + 1)) / 2 ** n
    upper = sum(math.comb(n, k) for k in range(positive, n + 1)) / 2 ** n
    return min(1.0, 2 * min(lower, upper))


def fmt(value):
    return "" if not math.isfinite(value) else f"{value:.6f}"


def write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def aggregate_test_conditions(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["family"], row["instance"], row["test_seed"])].append(row)
    result = []
    for key, items in sorted(groups.items()):
        result.append({
            "method": key[0], "family": key[1], "instance": key[2], "test_seed": key[3],
            "n_training_checkpoints": len(items),
            "clean": mean([item["clean"] for item in items]),
            "event": mean([item["event"] for item in items]),
            "delta": mean([item["delta"] for item in items]),
            "relative": mean([item["relative"] for item in items]),
        })
    return result


def within_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["family"])].append(row)
    output = []
    for (method, family), items in sorted(groups.items()):
        delta = [item["delta"] for item in items]
        relative = [item["relative"] for item in items]
        dl, dh = bootstrap(delta, 1100 + len(output))
        rl, rh = bootstrap(relative, 2100 + len(output))
        output.append({
            "method": method, "family": family, "n_raw_pairs": len(items),
            "n_test_conditions": len({(x["instance"], x["test_seed"]) for x in items}),
            "n_training_seeds": len({training_seed(x) for x in items}),
            "clean_mean": fmt(mean([item["clean"] for item in items])),
            "clean_std": fmt(std([item["clean"] for item in items])),
            "event_mean": fmt(mean([item["event"] for item in items])),
            "event_std": fmt(std([item["event"] for item in items])),
            "delta_mean": fmt(mean(delta)), "delta_std": fmt(std(delta)),
            "delta_ci95_low": fmt(dl), "delta_ci95_high": fmt(dh),
            "relative_mean": fmt(mean(relative)), "relative_std": fmt(std(relative)),
            "relative_ci95_low": fmt(rl), "relative_ci95_high": fmt(rh),
            "positive_delta_rate": fmt(sum(x > 0 for x in delta) / len(delta)),
            "delta_sign_test_p": fmt(sign_pvalue(delta)),
        })
    return output


def pairwise(conditions: list[dict], reference: str) -> list[dict]:
    by_method = defaultdict(dict)
    for row in conditions:
        by_method[row["method"]][(row["family"], row["instance"], row["test_seed"])] = row
    ref = by_method[reference]
    output = []
    for method, mapping in sorted(by_method.items()):
        if method == reference:
            continue
        common = sorted(set(ref) & set(mapping))
        for family in sorted({key[0] for key in common}):
            keys = [key for key in common if key[0] == family]
            differences = {field: [mapping[key][field] - ref[key][field] for key in keys]
                           for field in ("clean", "event", "delta", "relative")}
            row = {"reference_method": reference, "method": method, "family": family, "n_test_conditions": len(keys)}
            for index, field in enumerate(("clean", "event", "delta", "relative")):
                values = differences[field]
                low, high = bootstrap(values, 3100 + index + len(output))
                row[f"{field}_difference_mean"] = fmt(mean(values))
                row[f"{field}_difference_ci95_low"] = fmt(low)
                row[f"{field}_difference_ci95_high"] = fmt(high)
                row[f"{field}_difference_sign_test_p"] = fmt(sign_pvalue(values))
            output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True, help="LABEL=paired.csv")
    parser.add_argument("--within-output", required=True)
    parser.add_argument("--condition-output", required=True)
    parser.add_argument("--comparison-output", required=True)
    parser.add_argument("--reference-method", default="本文方法")
    args = parser.parse_args()
    all_rows = []
    for spec in args.method:
        _, rows = read_rows(spec)
        for row in rows:
            row["training_seed"] = training_seed(row)
        all_rows.extend(rows)
    conditions = aggregate_test_conditions(all_rows)
    write(Path(args.within_output), within_summary(all_rows))
    write(Path(args.condition_output), conditions)
    write(Path(args.comparison_output), pairwise(conditions, args.reference_method))
    print(f"within={args.within_output}")
    print(f"conditions={args.condition_output}")
    print(f"comparison={args.comparison_output}")


if __name__ == "__main__":
    main()
