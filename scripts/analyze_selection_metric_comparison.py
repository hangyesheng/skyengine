#!/usr/bin/env python3
"""Compare validation-gap and validation-Cmax model selection on paired tests.

This analysis is deliberately separate from the frozen thesis main tables.  It
reads the pre-registered paired-selection benchmark directories and produces
descriptive and paired statistics without changing any model or test result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


NAME_RE = re.compile(
    r"^paired_selection_(?P<variant>v1|cmax)_seed(?P<train_seed>\d+)"
    r"_(?P<scenario>base|dynamic)_runs3_t601_20261013$"
)


def load_rows(root: Path) -> dict[tuple, dict]:
    rows = {}
    for result_path in root.glob("paired_selection_*_20261013/results.jsonl"):
        match = NAME_RE.match(result_path.parent.name)
        if not match:
            continue
        metadata = match.groupdict()
        for line in result_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (
                metadata["scenario"],
                int(metadata["train_seed"]),
                record["family"],
                record["instance"],
                int(record["run"]),
                int(record.get("seed", -1)),
            )
            rows[(metadata["variant"],) + key] = {
                **record,
                "variant": metadata["variant"],
                "scenario": metadata["scenario"],
                "train_seed": int(metadata["train_seed"]),
            }
    return rows


def paired_rows(rows: dict[tuple, dict], scenario: str, train_seed: int | None = None):
    v1 = {}
    cmax = {}
    for (variant, row_scenario, row_seed, family, instance, run, test_seed), row in rows.items():
        if row_scenario != scenario or (train_seed is not None and row_seed != train_seed):
            continue
        key = (row_seed, family, instance, run, test_seed)
        if variant == "v1":
            v1[key] = row
        else:
            cmax[key] = row
    result = []
    for key in sorted(v1.keys() & cmax.keys()):
        left, right = v1[key], cmax[key]
        result.append({
            "scenario": scenario,
            "train_seed": key[0],
            "family": key[1],
            "instance": key[2],
            "run": key[3],
            "test_seed": key[4],
            "v1_makespan": left["makespan"],
            "cmax_makespan": right["makespan"],
            "delta_cmax_minus_v1": right["makespan"] - left["makespan"],
        })
    return result


def bootstrap_ci(values: list[float], seed: int, samples: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = [mean(rng.choices(values, k=n)) for _ in range(samples)]
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def sign_test_pvalue(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = sum(value < 0 for value in nonzero)
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("benchmark_results"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmark_plots/selection_metric_test_summary_v1.csv"),
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=Path("benchmark_plots/selection_metric_paired_differences_v1.csv"),
    )
    parser.add_argument(
        "--statistics-output",
        type=Path,
        default=Path("benchmark_plots/selection_metric_paired_statistics_v1.csv"),
    )
    args = parser.parse_args()

    rows = load_rows(args.root)
    paired = []
    summary = []
    statistics_rows = []
    for scenario in ("base", "dynamic"):
        for train_seed in (101, 102, 103, 104, 105):
            group = paired_rows(rows, scenario, train_seed)
            paired.extend(group)
            for variant in ("v1", "cmax"):
                values = [
                    row["v1_makespan"] if variant == "v1" else row["cmax_makespan"]
                    for row in group
                ]
                summary.append({
                    "scenario": scenario,
                    "train_seed": train_seed,
                    "variant": variant,
                    "n": len(values),
                    "mean_makespan": mean(values) if values else float("nan"),
                    "median_makespan": median(values) if values else float("nan"),
                    "std_makespan": stdev(values) if len(values) > 1 else 0.0,
                })

        group = paired_rows(rows, scenario)
        deltas = [row["delta_cmax_minus_v1"] for row in group]
        v1_values = [row["v1_makespan"] for row in group]
        cmax_values = [row["cmax_makespan"] for row in group]
        ci_low, ci_high = bootstrap_ci(deltas, 601013 + (0 if scenario == "base" else 1))
        summary_stats = {
            "scenario": scenario,
            "train_seed": "all",
            "n": len(deltas),
            "v1_mean_makespan": mean(v1_values) if v1_values else float("nan"),
            "cmax_mean_makespan": mean(cmax_values) if cmax_values else float("nan"),
            "mean_delta_cmax_minus_v1": mean(deltas) if deltas else float("nan"),
            "median_delta_cmax_minus_v1": median(deltas) if deltas else float("nan"),
            "relative_change": (mean(cmax_values) - mean(v1_values)) / mean(v1_values)
            if v1_values else float("nan"),
            "wins_cmax": sum(delta < 0 for delta in deltas),
            "losses_cmax": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "bootstrap95_low": ci_low,
            "bootstrap95_high": ci_high,
            "sign_test_pvalue": sign_test_pvalue(deltas),
        }
        statistics_rows.append(summary_stats)

    write_csv(args.summary_output, summary)
    write_csv(args.paired_output, paired)
    write_csv(args.statistics_output, statistics_rows)
    print(f"wrote {len(summary)} summary rows, {len(paired)} paired rows, {len(statistics_rows)} statistics rows")


if __name__ == "__main__":
    main()
