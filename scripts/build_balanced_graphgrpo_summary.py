#!/usr/bin/env python3
"""Build a balanced, auditable summary for the five frozen model seeds.

The benchmark runner stores a derived trial seed in each JSONL record.  This
script maps that value back to the explicit test benchmark seed (601--603),
validates every base/event pair, and keeps the training checkpoint seed as a
separate factor.  It writes pair-level, training-seed, instance-level and
bootstrap/sign-test summaries for the frozen main method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"


MODEL_BATCHES = {
    101: [
        ("graphgrpo_frozen_lb_test_seed601_clean_v2_20261012", "graphgrpo_frozen_lb_test_seed601_uncertain_v2_20261012"),
        ("graphgrpo_frozen_lb_test_model101_seed602_clean_v3_20261012", "graphgrpo_frozen_lb_test_model101_seed602_uncertain_v3_20261012"),
        ("graphgrpo_frozen_lb_test_model101_seed603_clean_v3_20261012", "graphgrpo_frozen_lb_test_model101_seed603_uncertain_v3_20261012"),
    ],
    102: [
        ("graphgrpo_frozen_lb_test_m102_t601_clean_v3_20261012", "graphgrpo_frozen_lb_test_m102_t601_event_v3_20261012"),
        ("graphgrpo_frozen_lb_test_seed602_clean_v2_20261012", "graphgrpo_frozen_lb_test_seed602_uncertain_v2_20261012"),
        ("graphgrpo_frozen_lb_test_m102_t603_clean_v3_20261012", "graphgrpo_frozen_lb_test_m102_t603_event_v3_20261012"),
    ],
    103: [
        ("graphgrpo_frozen_lb_test_m103_t601_clean_v3_20261012", "graphgrpo_frozen_lb_test_m103_t601_event_v3_20261012"),
        ("graphgrpo_frozen_lb_test_m103_t602_clean_v3_20261012", "graphgrpo_frozen_lb_test_m103_t602_event_v3_20261012"),
        ("graphgrpo_frozen_lb_test_seed603_clean_v2_20261012", "graphgrpo_frozen_lb_test_seed603_uncertain_v2_20261012"),
    ],
    104: [
        ("graphgrpo_frozen_lb_test_seed104_clean_v2_20261012", "graphgrpo_frozen_lb_test_seed104_uncertain_v2_20261012"),
    ],
    105: [
        ("graphgrpo_frozen_lb_test_seed105_clean_v2_20261012", "graphgrpo_frozen_lb_test_seed105_uncertain_v2_20261012"),
    ],
}


def derive_trial_seed(base_seed: int, family: str, instance: str, run: int) -> int:
    identity = f"{base_seed}:{family}:{instance}:{run}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def load_records(experiment_id: str) -> list[dict]:
    path = RESULTS_ROOT / experiment_id / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def event_names(values) -> list[str]:
    names = {str(value).rsplit(".", 1)[-1].strip() for value in (values or [])}
    return sorted(name for name in names if name)


def read_map(experiment_id: str) -> dict[tuple, dict]:
    records = load_records(experiment_id)
    result = {}
    for record in records:
        key = (record.get("agent"), record.get("family"), record.get("instance"),
               record.get("run"), record.get("seed"))
        if key in result:
            raise ValueError(f"{experiment_id}: duplicate key {key}")
        result[key] = record
    return result


def identify_test_seed(record: dict) -> int:
    family, instance, run, actual = record.get("family"), record.get("instance"), record.get("run"), record.get("seed")
    candidates = [seed for seed in (601, 602, 603) if derive_trial_seed(seed, family, instance, run) == actual]
    if len(candidates) != 1:
        raise ValueError(f"cannot identify test seed for {family}/{instance}/run{run}: {actual} -> {candidates}")
    return candidates[0]


def collect_pairs() -> list[dict]:
    rows = []
    for training_seed, batches in MODEL_BATCHES.items():
        seen_test_seeds = set()
        model_hashes = set()
        for clean_id, event_id in batches:
            clean = read_map(clean_id)
            event = read_map(event_id)
            if set(clean) != set(event):
                raise ValueError(f"{clean_id}/{event_id}: clean/event keys differ")
            for key in sorted(clean):
                base, dynamic = clean[key], event[key]
                test_seed = identify_test_seed(base)
                seen_test_seeds.add(test_seed)
                if base.get("seed") != dynamic.get("seed"):
                    raise ValueError(f"{clean_id}/{event_id}/{key}: derived seed differs")
                if base.get("model_sha256") != dynamic.get("model_sha256"):
                    raise ValueError(f"{clean_id}/{event_id}/{key}: model hash differs")
                if not base.get("model_loaded") or not dynamic.get("model_loaded"):
                    raise ValueError(f"{clean_id}/{event_id}/{key}: model_loaded is false")
                if dynamic.get("events_loaded") is not True:
                    raise ValueError(f"{event_id}/{key}: events_loaded is not true")
                planned = int(dynamic.get("planned_event_count") or 0)
                executed = int(dynamic.get("executed_event_count") or 0)
                if planned != executed:
                    raise ValueError(f"{event_id}/{key}: planned={planned}, executed={executed}")
                if base.get("status") != "completed" or dynamic.get("status") != "completed":
                    raise ValueError(f"{clean_id}/{event_id}/{key}: incomplete status")
                if not finite(base.get("makespan")) or not finite(dynamic.get("makespan")):
                    raise ValueError(f"{clean_id}/{event_id}/{key}: invalid makespan")
                names = event_names(dynamic.get("executed_event_types"))
                rows.append({
                    "training_seed": training_seed,
                    "test_seed": test_seed,
                    "agent": key[0], "family": key[1], "instance": key[2], "run": key[3],
                    "clean_makespan": float(base["makespan"]),
                    "event_makespan": float(dynamic["makespan"]),
                    "delta_makespan": float(dynamic["makespan"]) - float(base["makespan"]),
                    "relative_degradation": (float(dynamic["makespan"]) - float(base["makespan"])) / float(base["makespan"]),
                    "planned_event_count": planned,
                    "executed_event_count": executed,
                    "event_signature": "+".join(names) if names else "NONE",
                    "clean_experiment_id": clean_id,
                    "event_experiment_id": event_id,
                    "model_sha256": base.get("model_sha256", ""),
                    "clean_decision_time": (base.get("decision_stats") or {}).get("average_decision_time"),
                    "event_decision_time": (dynamic.get("decision_stats") or {}).get("average_decision_time"),
                })
                model_hashes.add(base.get("model_sha256"))
        if seen_test_seeds != {601, 602, 603}:
            raise ValueError(f"training seed {training_seed}: expected test seeds 601/602/603, got {sorted(seen_test_seeds)}")
        if len(model_hashes) != 1:
            raise ValueError(f"training seed {training_seed}: multiple checkpoint hashes {model_hashes}")
    if len(rows) != 90:
        raise ValueError(f"expected 90 pairs, got {len(rows)}")
    return rows


def mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    avg = sum(values) / len(values)
    sd = math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))
    return f"{avg:.6f}", f"{sd:.6f}"


def median(values: list[float]) -> str:
    if not values:
        return ""
    values = sorted(values)
    middle = len(values) // 2
    value = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return f"{value:.6f}"


def exact_sign_pvalue(values: list[float]) -> str:
    nonzero = [value for value in values if value != 0]
    n = len(nonzero)
    if not n:
        return "1.000000"
    positive = sum(value > 0 for value in nonzero)
    tail = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2 ** n)
    p = min(1.0, 2 * min(tail, 1 - tail + math.comb(n, positive) / (2 ** n)))
    return f"{p:.6f}"


def bootstrap(values: list[float], seed: int = 20261012, n_boot: int = 10000) -> tuple[str, str]:
    if not values:
        return "", ""
    # Local deterministic linear-congruential generator avoids a dependency on
    # NumPy for artifact generation.
    state = seed & 0xFFFFFFFF
    samples = []
    count = len(values)
    for _ in range(n_boot):
        total = 0.0
        for _ in range(count):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            total += values[state % count]
        samples.append(total / count)
    samples.sort()
    low = samples[int(0.025 * n_boot)]
    high = samples[int(0.975 * n_boot) - 1]
    return f"{low:.6f}", f"{high:.6f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, items in sorted(groups.items()):
        clean = [item["clean_makespan"] for item in items]
        event = [item["event_makespan"] for item in items]
        delta = [item["delta_makespan"] for item in items]
        relative = [item["relative_degradation"] for item in items]
        cm, cs = mean_std(clean); em, es = mean_std(event); dm, ds = mean_std(delta); rm, rs = mean_std(relative)
        output.append({
            **dict(zip(keys, key)), "n_pairs": len(items),
            "clean_mean": cm, "clean_std": cs, "clean_median": median(clean),
            "event_mean": em, "event_std": es, "event_median": median(event),
            "delta_mean": dm, "delta_std": ds, "delta_median": median(delta),
            "relative_mean": rm, "relative_std": rs, "relative_median": median(relative),
            "positive_delta_rate": f"{sum(value > 0 for value in delta) / len(delta):.6f}",
            "planned_events": sum(item["planned_event_count"] for item in items),
            "executed_events": sum(item["executed_event_count"] for item in items),
        })
    return output


def statistical_rows(rows: list[dict]) -> list[dict]:
    groups = [("all", rows)]
    groups.extend((family, [row for row in rows if row["family"] == family]) for family in sorted({row["family"] for row in rows}))
    output = []
    for label, items in groups:
        delta = [row["delta_makespan"] for row in items]
        relative = [row["relative_degradation"] for row in items]
        dl, dh = bootstrap(delta); rl, rh = bootstrap(relative, seed=20261013)
        output.append({
            "group": label, "n_pairs": len(items),
            "delta_mean": f"{mean(delta):.6f}", "delta_bootstrap_low": dl, "delta_bootstrap_high": dh,
            "relative_mean": f"{mean(relative):.6f}", "relative_bootstrap_low": rl, "relative_bootstrap_high": rh,
            "positive_delta_count": sum(value > 0 for value in delta),
            "zero_delta_count": sum(value == 0 for value in delta),
            "negative_delta_count": sum(value < 0 for value in delta),
            "positive_delta_rate": f"{sum(value > 0 for value in delta) / len(delta):.6f}",
            "exact_sign_test_p": exact_sign_pvalue(delta),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmark_plots")
    args = parser.parse_args()
    rows = collect_pairs()
    output_dir = Path(args.output_dir)
    pair_path = output_dir / "formal_graphgrpo_frozen_lb_five_seed_pairs_v1.csv"
    train_path = output_dir / "formal_graphgrpo_frozen_lb_five_seed_by_training_seed_v1.csv"
    instance_path = output_dir / "formal_graphgrpo_frozen_lb_five_seed_by_instance_v1.csv"
    stats_path = output_dir / "formal_graphgrpo_frozen_lb_five_seed_paired_statistics_v1.csv"
    write_csv(pair_path, rows)
    write_csv(train_path, aggregate(rows, ("training_seed", "family")))
    write_csv(instance_path, aggregate(rows, ("family", "instance")))
    write_csv(stats_path, statistical_rows(rows))
    print(f"pairs={pair_path} n={len(rows)}")
    print(f"training_summary={train_path}")
    print(f"instance_summary={instance_path}")
    print(f"statistics={stats_path}")


if __name__ == "__main__":
    main()
