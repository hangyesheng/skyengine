#!/usr/bin/env python3
"""Analyze the cross-family generalization benchmark.

The benchmark has one OR-Tools and one HGS run for each test condition, while
the learning method is evaluated with several independently trained
checkpoints.  The script first averages learning checkpoints at each
``(family, instance, test_seed, scenario)`` condition, then performs paired
comparisons between methods.  It also verifies that every included trial was
completed and that dynamic-event records executed the planned event list.

The output is deliberately separate from the frozen main-test tables.  It is
intended for generalization analysis and for deciding whether the candidate
checkpoint-selection rule is ready for a formal experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


EXPECTED_FAMILIES = {"dauzere", "fattahi", "barnes", "behnke"}
EXPECTED_INSTANCES = {
    "dauzere": {"01a", "02a", "03a", "04a", "05a"},
    "fattahi": {"mfjs01", "mfjs02", "mfjs03", "mfjs04", "mfjs05"},
    "barnes": {"mt10c1", "mt10cc", "mt10x", "mt10xx", "mt10xxx"},
    "behnke": {"lar01_1", "lar01_2", "lar01_3", "lar01_4", "lar01_5"},
}


def derive_trial_seed(base_seed: int, family: str, instance: str, run: int) -> int:
    identity = f"{base_seed}:{family}:{instance}:{run}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def read_jsonl(path: Path, method: str, train_seed: str, scenario: str, test_seed: int) -> list[dict]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "completed":
            raise ValueError(f"{path}:{line_no}: non-completed status {row.get('status')}")
        if row.get("makespan") is None or float(row["makespan"]) <= 0:
            raise ValueError(f"{path}:{line_no}: invalid makespan")
        if int(row.get("seed")) != derive_trial_seed(test_seed, row["family"], row["instance"], int(row["run"])):
            raise ValueError(f"{path}:{line_no}: trial seed does not match test seed {test_seed}")
        if scenario == "dynamic":
            planned = int(row.get("planned_event_count") or 0)
            executed = int(row.get("executed_event_count") or 0)
            if not row.get("events_loaded", False):
                raise ValueError(f"{path}:{line_no}: dynamic event file was not loaded")
            if planned != executed:
                raise ValueError(f"{path}:{line_no}: planned={planned}, executed={executed}")
        rows.append({
            "method": method,
            "train_seed": train_seed,
            "scenario": scenario,
            "test_seed": test_seed,
            "family": row["family"],
            "instance": row["instance"],
            "run": int(row["run"]),
            "makespan": float(row["makespan"]),
            "planned_event_count": int(row.get("planned_event_count") or 0),
            "executed_event_count": int(row.get("executed_event_count") or 0),
            "executed_event_types": ";".join(sorted(row.get("executed_event_types") or [])),
            "source": str(path.parent.name),
        })
    return rows


def collect_rows(root: Path, graph_prefix: str = "generalization_graphgrpo_cmax", method_label: str = "本文方法") -> list[dict]:
    rows = []
    # The corrected seed-103/603 directories supersede the two audit-only
    # directories containing the intentionally mistyped Behnke instance.
    for test_seed in (601, 602, 603):
        for scenario in ("base", "dynamic"):
            for train_seed in (101, 102, 103):
                suffix = f"_{scenario}_t{test_seed}_20261013"
                if graph_prefix.endswith("cmax") and train_seed == 103 and test_seed == 603:
                    suffix = f"_{scenario}_t{test_seed}_corrected_20261013"
                path = root / f"{graph_prefix}_seed{train_seed}{suffix}" / "results.jsonl"
                if path.exists():
                    rows.extend(read_jsonl(path, method_label, str(train_seed), scenario, test_seed))

            hgs = root / f"generalization_hgs_seed101_{scenario}_t{test_seed}_20261013" / "results.jsonl"
            ort = root / f"generalization_ortools_{scenario}_t{test_seed}_20261013" / "results.jsonl"
            rows.extend(read_jsonl(hgs, "HGS适配实现", "none", scenario, test_seed))
            rows.extend(read_jsonl(ort, "OR-Tools", "none", scenario, test_seed))

    expected_keys = {(family, instance) for family, instances in EXPECTED_INSTANCES.items() for instance in instances}
    for method in (method_label, "HGS适配实现", "OR-Tools"):
        for scenario in ("base", "dynamic"):
            for test_seed in (601, 602, 603):
                subset = [r for r in rows if r["method"] == method and r["scenario"] == scenario and r["test_seed"] == test_seed]
                keys = {(r["family"], r["instance"]) for r in subset}
                if keys != expected_keys:
                    raise ValueError(f"{method}/{scenario}/t{test_seed}: expected 20 instances, got {len(keys)}")
                # Learning methods deliberately have one row per training
                # checkpoint; the checkpoint identifier is part of the raw
                # audit key.  The later aggregation removes that replication
                # before method-to-method comparisons.
                seen = [(r["train_seed"], r["family"], r["instance"], r["run"]) for r in subset]
                if len(seen) != len(set(seen)):
                    raise ValueError(f"duplicate trial keys for {method}/{scenario}/t{test_seed}")
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * probability
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (pos - low)


def bootstrap_ci(values: list[float], seed: int, repetitions: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    samples = [mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(repetitions)]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def sign_pvalue(values: list[float]) -> float:
    values = [v for v in values if v != 0]
    if not values:
        return 1.0
    n = len(values)
    positives = sum(v > 0 for v in values)
    lower = sum(math.comb(n, k) for k in range(positives + 1)) / (2 ** n)
    upper = sum(math.comb(n, k) for k in range(positives, n + 1)) / (2 ** n)
    return min(1.0, 2 * min(lower, upper))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_conditions(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["scenario"], row["test_seed"], row["family"], row["instance"], row["run"])].append(row)
    result = []
    for key, items in sorted(grouped.items()):
        result.append({
            "method": key[0], "scenario": key[1], "test_seed": key[2],
            "family": key[3], "instance": key[4], "run": key[5],
            "n_training_checkpoints": len(items),
            "training_seeds": ";".join(sorted(item["train_seed"] for item in items if item["train_seed"] != "none")),
            "makespan": mean([item["makespan"] for item in items]),
            "planned_event_count": mean([item["planned_event_count"] for item in items]),
            "executed_event_count": mean([item["executed_event_count"] for item in items]),
        })
    return result


def summarize_conditions(conditions: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in conditions:
        grouped[(row["method"], row["scenario"], row["family"])].append(row)
    output = []
    for index, ((method, scenario, family), items) in enumerate(sorted(grouped.items())):
        values = [item["makespan"] for item in items]
        low, high = bootstrap_ci(values, 1000 + index)
        output.append({
            "method": method, "scenario": scenario, "family": family,
            "n_conditions": len(values), "n_test_seeds": len({item["test_seed"] for item in items}),
            "n_instances": len({item["instance"] for item in items}),
            "makespan_mean": mean(values), "makespan_std": std(values),
            "makespan_ci95_low": low, "makespan_ci95_high": high,
            "mean_training_checkpoints": mean([item["n_training_checkpoints"] for item in items]),
            "planned_event_mean": mean([item["planned_event_count"] for item in items]),
            "executed_event_mean": mean([item["executed_event_count"] for item in items]),
        })
    return output


def pairwise(conditions: list[dict], reference: str = "本文方法") -> list[dict]:
    by_method = defaultdict(dict)
    for row in conditions:
        key = (row["scenario"], row["test_seed"], row["family"], row["instance"], row["run"])
        by_method[row["method"]][key] = row
    output = []
    ref = by_method[reference]
    for method in sorted(by_method):
        if method == reference:
            continue
        common = sorted(set(ref) & set(by_method[method]))
        for scenario in ("base", "dynamic"):
            for family in [*sorted(EXPECTED_FAMILIES), "ALL"]:
                keys = [key for key in common if key[0] == scenario and key[2] == family]
                if family == "ALL":
                    keys = [key for key in common if key[0] == scenario]
                if not keys:
                    continue
                diffs = [by_method[method][key]["makespan"] - ref[key]["makespan"] for key in keys]
                low, high = bootstrap_ci(diffs, 3000 + len(output))
                output.append({
                    "reference_method": reference, "method": method, "scenario": scenario,
                    "family": family, "n_conditions": len(diffs),
                    "baseline_minus_reference_mean": mean(diffs),
                    "baseline_minus_reference_std": std(diffs),
                    "bootstrap95_low": low, "bootstrap95_high": high,
                    "baseline_wins": sum(d > 0 for d in diffs),
                    "baseline_losses": sum(d < 0 for d in diffs),
                    "ties": sum(d == 0 for d in diffs),
                    "sign_test_pvalue": sign_pvalue(diffs),
                })
    return output


def degradation(conditions: list[dict]) -> list[dict]:
    by_key = {(r["method"], r["test_seed"], r["family"], r["instance"], r["run"], r["scenario"]): r for r in conditions}
    paired = []
    for method in ("本文方法", "HGS适配实现", "OR-Tools"):
        keys = {(r["test_seed"], r["family"], r["instance"], r["run"])
                for r in conditions if r["method"] == method and r["scenario"] == "base"}
        for test_seed, family, instance, run in sorted(keys):
            base = by_key[(method, test_seed, family, instance, run, "base")]
            dynamic = by_key[(method, test_seed, family, instance, run, "dynamic")]
            paired.append({
                "method": method, "test_seed": test_seed, "family": family,
                "instance": instance, "run": run,
                "base_makespan": base["makespan"], "dynamic_makespan": dynamic["makespan"],
                "delta_makespan": dynamic["makespan"] - base["makespan"],
                "relative_degradation": (dynamic["makespan"] - base["makespan"]) / base["makespan"],
                "n_training_checkpoints": dynamic["n_training_checkpoints"],
            })
    return paired


def summarize_degradation(paired: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["method"], row["family"])].append(row)
    output = []
    for index, ((method, family), items) in enumerate(sorted(grouped.items())):
        deltas = [item["delta_makespan"] for item in items]
        relatives = [item["relative_degradation"] for item in items]
        dl, dh = bootstrap_ci(deltas, 5000 + index)
        rl, rh = bootstrap_ci(relatives, 6000 + index)
        output.append({
            "method": method, "family": family, "n_pairs": len(items),
            "n_test_seeds": len({item["test_seed"] for item in items}),
            "base_mean": mean([item["base_makespan"] for item in items]),
            "dynamic_mean": mean([item["dynamic_makespan"] for item in items]),
            "delta_mean": mean(deltas), "delta_std": std(deltas),
            "delta_ci95_low": dl, "delta_ci95_high": dh,
            "relative_mean": mean(relatives), "relative_std": std(relatives),
            "relative_ci95_low": rl, "relative_ci95_high": rh,
            "positive_degradation_rate": sum(d > 0 for d in deltas) / len(deltas),
            "delta_sign_test_pvalue": sign_pvalue(deltas),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output-prefix", type=Path, default=Path("benchmark_plots/generalization_v1"))
    parser.add_argument("--graph-prefix", default="generalization_graphgrpo_cmax", help="learning-result directory prefix")
    parser.add_argument("--method-label", default="本文方法")
    args = parser.parse_args()

    rows = collect_rows(args.root, args.graph_prefix, args.method_label)
    conditions = aggregate_conditions(rows)
    summaries = summarize_conditions(conditions)
    comparisons = pairwise(conditions)
    degradation_rows = degradation(conditions)
    degradation_summary = summarize_degradation(degradation_rows)
    prefix = args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_raw_audit.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_conditions.csv"), conditions)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summaries)
    write_csv(prefix.with_name(prefix.name + "_comparisons.csv"), comparisons)
    write_csv(prefix.with_name(prefix.name + "_degradation.csv"), degradation_rows)
    write_csv(prefix.with_name(prefix.name + "_degradation_summary.csv"), degradation_summary)
    print(f"raw={len(rows)} conditions={len(conditions)} summaries={len(summaries)} comparisons={len(comparisons)} degradation={len(degradation_rows)}")


if __name__ == "__main__":
    main()
