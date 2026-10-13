#!/usr/bin/env python3
"""Summarize the formal Cmax-aligned checkpoint comparison.

The script joins the three Cmax-selected learning checkpoints with the HGS
adaptation and OR-Tools runs on the same six held-out instances, test seeds,
and event timelines.  Learning checkpoints are averaged within each test
condition before comparisons so that the single OR-Tools run is not counted
three times.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


EXPECTED = {"brandimarte": {"mk11", "mk12", "mk13", "mk14", "mk15"}, "kacem": {"k4"}}


def trial_seed(base: int, family: str, instance: str, run: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{family}:{instance}:{run}".encode()).digest()[:4], "big")


def read(path: Path, method: str, train_seed: str, scenario: str, test_seed: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    result = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "completed" or row.get("makespan") is None:
            raise ValueError(f"{path}:{line_no}: incomplete trial")
        if int(row.get("seed")) != trial_seed(test_seed, row["family"], row["instance"], int(row["run"])):
            raise ValueError(f"{path}:{line_no}: trial seed mismatch")
        if scenario == "dynamic":
            if not row.get("events_loaded", False):
                raise ValueError(f"{path}:{line_no}: event file not loaded")
            if int(row.get("planned_event_count") or 0) != int(row.get("executed_event_count") or 0):
                raise ValueError(f"{path}:{line_no}: planned/executed event mismatch")
        result.append({
            "method": method, "train_seed": train_seed, "scenario": scenario,
            "test_seed": test_seed, "family": row["family"], "instance": row["instance"],
            "run": int(row["run"]), "makespan": float(row["makespan"]),
            "planned_event_count": int(row.get("planned_event_count") or 0),
            "executed_event_count": int(row.get("executed_event_count") or 0),
        })
    return result


def collect(root: Path) -> list[dict]:
    rows = []
    for test_seed in (601, 602, 603):
        for scenario in ("base", "dynamic"):
            graph_scenario = scenario
            for train_seed in (101, 102, 103):
                graph = root / f"formal_cmax_graphgrpo_seed{train_seed}_{graph_scenario}_t{test_seed}_20261013" / "results.jsonl"
                rows.extend(read(graph, "本文方法", str(train_seed), scenario, test_seed))

                # The archived HGS benchmark uses the final checkpoint for
                # one diagonal test seed and explicit cross-seed directories
                # for the other two conditions.
                diagonal = {601: 101, 602: 102, 603: 103}[test_seed]
                if train_seed == diagonal:
                    hgs_name = f"hgs_formal_final_seed{train_seed}_test_{'base' if scenario == 'base' else 'event'}_v1_20261011"
                else:
                    hgs_name = f"hgs_formal_cross_m{train_seed}_t{test_seed}_{'base' if scenario == 'base' else 'event'}_v1_20261012"
                hgs = root / hgs_name / "results.jsonl"
                # HGS is loaded once per checkpoint; the loop is intentional
                # because each checkpoint is a paired training condition.
                rows.extend(read(hgs, "HGS适配实现", str(train_seed), scenario, test_seed))

            if test_seed == 601:
                ort_name = f"ortools_formal_seed601_test_{'clean' if scenario == 'base' else 'uncertain'}_v1_20261010"
            else:
                ort_name = f"ortools_formal_cross_t{test_seed}_{'base' if scenario == 'base' else 'event'}_v1_20261012"
            ort = root / ort_name / "results.jsonl"
            rows.extend(read(ort, "OR-Tools", "none", scenario, test_seed))

    expected = {(f, i) for f, instances in EXPECTED.items() for i in instances}
    for method in ("本文方法", "HGS适配实现", "OR-Tools"):
        for scenario in ("base", "dynamic"):
            for seed in (601, 602, 603):
                subset = [r for r in rows if r["method"] == method and r["scenario"] == scenario and r["test_seed"] == seed]
                if {(r["family"], r["instance"]) for r in subset} != expected:
                    raise ValueError(f"{method}/{scenario}/t{seed}: expected six test instances")
    return rows


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def std(x):
    if not x:
        return float("nan")
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def percentile(x, p):
    x = sorted(x)
    if not x:
        return float("nan")
    pos = (len(x) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[lo] if lo == hi else x[lo] + (x[hi] - x[lo]) * (pos - lo)


def bootstrap(x, seed, n=10000):
    if not x:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = [mean([x[rng.randrange(len(x))] for _ in x]) for _ in range(n)]
    return percentile(samples, .025), percentile(samples, .975)


def sign_p(x):
    x = [v for v in x if v]
    if not x:
        return 1.0
    n = len(x)
    k = sum(v > 0 for v in x)
    tail = min(sum(math.comb(n, i) for i in range(k + 1)), sum(math.comb(n, i) for i in range(k, n + 1))) / 2**n
    return min(1.0, 2 * tail)


def write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["scenario"], row["test_seed"], row["family"], row["instance"], row["run"])].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        out.append({
            "method": key[0], "scenario": key[1], "test_seed": key[2], "family": key[3], "instance": key[4], "run": key[5],
            "n_checkpoints": len(items), "makespan": mean([r["makespan"] for r in items]),
            "planned_event_count": mean([r["planned_event_count"] for r in items]), "executed_event_count": mean([r["executed_event_count"] for r in items]),
        })
    return out


def summaries(conditions):
    grouped = defaultdict(list)
    for row in conditions:
        grouped[(row["method"], row["scenario"])].append(row)
    out = []
    for idx, ((method, scenario), items) in enumerate(sorted(grouped.items())):
        v = [r["makespan"] for r in items]; lo, hi = bootstrap(v, 100 + idx)
        out.append({"method": method, "scenario": scenario, "n_conditions": len(v), "n_test_seeds": len({r["test_seed"] for r in items}), "makespan_mean": mean(v), "makespan_std": std(v), "makespan_ci95_low": lo, "makespan_ci95_high": hi})
    return out


def pairwise(conditions):
    by = defaultdict(dict)
    for r in conditions:
        by[r["method"]][(r["scenario"], r["test_seed"], r["family"], r["instance"], r["run"])] = r
    out = []
    for other in ("HGS适配实现", "OR-Tools"):
        for scenario in ("base", "dynamic"):
            keys = sorted(set(by["本文方法"]) & set(by[other]))
            keys = [k for k in keys if k[0] == scenario]
            d = [by[other][k]["makespan"] - by["本文方法"][k]["makespan"] for k in keys]
            lo, hi = bootstrap(d, 300 + len(out))
            out.append({"reference_method": "本文方法", "method": other, "scenario": scenario, "n_conditions": len(d), "baseline_minus_method_mean": mean(d), "baseline_minus_method_std": std(d), "bootstrap95_low": lo, "bootstrap95_high": hi, "baseline_wins": sum(v > 0 for v in d), "baseline_losses": sum(v < 0 for v in d), "sign_test_pvalue": sign_p(d)})
    return out


def degradation(conditions):
    by = {(r["method"], r["test_seed"], r["family"], r["instance"], r["run"], r["scenario"]): r for r in conditions}
    out = []
    for method in ("本文方法", "HGS适配实现", "OR-Tools"):
        keys = sorted({(r["test_seed"], r["family"], r["instance"], r["run"]) for r in conditions if r["method"] == method and r["scenario"] == "base"})
        for seed, family, instance, run in keys:
            b = by[(method, seed, family, instance, run, "base")]; d = by[(method, seed, family, instance, run, "dynamic")]
            delta = d["makespan"] - b["makespan"]
            out.append({"method": method, "test_seed": seed, "family": family, "instance": instance, "run": run, "base_makespan": b["makespan"], "dynamic_makespan": d["makespan"], "delta_makespan": delta, "relative_degradation": delta / b["makespan"]})
    return out


def degradation_summary(rows):
    grouped = defaultdict(list)
    for r in rows: grouped[(r["method"], r["family"])].append(r)
    out = []
    for idx, ((method, family), items) in enumerate(sorted(grouped.items())):
        d = [r["delta_makespan"] for r in items]; rel = [r["relative_degradation"] for r in items]; lo, hi = bootstrap(d, 500 + idx); rlo, rhi = bootstrap(rel, 600 + idx)
        out.append({"method": method, "family": family, "n_pairs": len(items), "delta_mean": mean(d), "delta_std": std(d), "delta_ci95_low": lo, "delta_ci95_high": hi, "relative_mean": mean(rel), "relative_ci95_low": rlo, "relative_ci95_high": rhi, "positive_degradation_rate": sum(v > 0 for v in d) / len(d), "delta_sign_test_pvalue": sign_p(d)})
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=Path("benchmark_results")); p.add_argument("--output-prefix", type=Path, default=Path("benchmark_plots/formal_cmax_v1")); a = p.parse_args()
    raw = collect(a.root); cond = aggregate(raw); pref = a.output_prefix
    write(pref.with_name(pref.name + "_raw.csv"), raw); write(pref.with_name(pref.name + "_conditions.csv"), cond); write(pref.with_name(pref.name + "_summary.csv"), summaries(cond)); write(pref.with_name(pref.name + "_comparisons.csv"), pairwise(cond)); deg = degradation(cond); write(pref.with_name(pref.name + "_degradation.csv"), deg); write(pref.with_name(pref.name + "_degradation_summary.csv"), degradation_summary(deg)); print(f"raw={len(raw)} conditions={len(cond)}")


if __name__ == "__main__": main()
