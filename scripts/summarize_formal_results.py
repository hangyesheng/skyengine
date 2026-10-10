#!/usr/bin/env python3
"""汇总多 seed 的正式 benchmark 结果。

该脚本把每个 benchmark 实验目录视为一个独立的测试批次，保留来源实验 ID，
避免把不同训练 checkpoint 的结果误合并成同一个 run。配对模式要求 clean 和
uncertain 实验 ID 一一对应，并按 ``(agent, family, instance, run)`` 在每一对
实验内部配对。
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"


def load_records(experiment_id: str):
    path = RESULTS_ROOT / experiment_id / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["source_experiment_id"] = experiment_id
            records.append(record)
    return records


def mean_std(values):
    """Descriptive mean and population standard deviation (ddof=0)."""
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return f"{mean:.6f}", f"{math.sqrt(variance):.6f}"


def median(values):
    if not values:
        return ""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        value = ordered[middle]
    else:
        value = (ordered[middle - 1] + ordered[middle]) / 2
    return f"{value:.6f}"


def event_signature(event_types):
    """Return a stable, namespace-free event signature and coarse category.

    Benchmark records currently store event names either as ``JOB_ADD`` or as
    fully-qualified names such as ``packet_factory.JOB_ADD``.  The summary
    should not depend on that implementation detail, so signatures are sorted
    and the namespace is removed before grouping.
    """
    names = set()
    for value in event_types or []:
        name = str(value).rsplit(".", 1)[-1].strip()
        if name:
            names.add(name)
    signature = "+".join(sorted(names)) if names else "NONE"
    category = signature if len(names) <= 1 else "MIXED"
    return signature, category


def write_status(records, output):
    grouped = defaultdict(int)
    for record in records:
        scenario = record.get("uncertain_scenario") or "clean"
        grouped[(record.get("source_experiment_id"), record.get("agent"),
                 record.get("family"), scenario, record.get("status", "unknown"))] += 1
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_experiment_id", "agent", "family", "scenario", "status", "n_records"])
        for key, count in sorted(grouped.items()):
            writer.writerow([*key, count])


def write_makespan_summary(records, output):
    grouped = defaultdict(list)
    for record in records:
        if record.get("status") != "completed" or record.get("makespan") is None:
            continue
        scenario = record.get("uncertain_scenario") or "clean"
        key = (record.get("agent"), record.get("family"), record.get("instance"), scenario)
        grouped[key].append(float(record["makespan"]))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", "family", "instance", "scenario", "n_trials",
                         "makespan_mean", "makespan_std", "makespan_median", "makespan_min", "makespan_max"])
        for key, values in sorted(grouped.items()):
            mean, std = mean_std(values)
            writer.writerow([*key, len(values), mean, std, median(values), f"{min(values):.6f}", f"{max(values):.6f}"])


def paired_rows(clean_ids, uncertain_ids):
    if len(clean_ids) != len(uncertain_ids):
        raise ValueError("clean 和 uncertain 实验 ID 数量必须相同")
    rows = []
    for clean_id, uncertain_id in zip(clean_ids, uncertain_ids):
        clean_records = load_records(clean_id)
        uncertain_records = load_records(uncertain_id)
        for records in (clean_records, uncertain_records):
            seen = set()
            for record in records:
                key = (record.get("agent"), record.get("family"), record.get("instance"), record.get("run"))
                if key in seen:
                    raise ValueError(f"实验 {record['source_experiment_id']} 有重复配对键: {key}")
                seen.add(key)
        clean_by_key = {
            (r.get("agent"), r.get("family"), r.get("instance"), r.get("run")): r
            for r in clean_records
            if r.get("status") == "completed" and r.get("makespan") is not None
        }
        uncertain_by_key = {
            (r.get("agent"), r.get("family"), r.get("instance"), r.get("run")): r
            for r in uncertain_records
            if r.get("status") == "completed" and r.get("makespan") is not None
        }
        for key in sorted(set(clean_by_key) & set(uncertain_by_key)):
            clean = clean_by_key[key]
            uncertain = uncertain_by_key[key]
            for field in ("seed", "model_sha256"):
                if clean.get(field) != uncertain.get(field):
                    raise ValueError(f"{clean_id}/{uncertain_id} 的 {key} 不满足 {field} 一致的配对协议")
            if uncertain.get("events_loaded") is False:
                raise ValueError(f"{uncertain_id} 的 {key} 未加载 uncertain 事件")
            clean_makespan = float(clean["makespan"])
            uncertain_makespan = float(uncertain["makespan"])
            if (not math.isfinite(clean_makespan) or clean_makespan <= 0
                    or not math.isfinite(uncertain_makespan) or uncertain_makespan < 0):
                raise ValueError(f"{clean_id}/{uncertain_id} 的 {key} 有无效 makespan")
            delta = uncertain_makespan - clean_makespan
            rows.append({
                "clean_experiment_id": clean_id,
                "uncertain_experiment_id": uncertain_id,
                "agent": key[0], "family": key[1], "instance": key[2], "run": key[3],
                "clean_makespan": clean_makespan,
                "uncertain_makespan": uncertain_makespan,
                "delta_makespan": delta,
                "relative_degradation": delta / clean_makespan if clean_makespan else None,
                "uncertain_scenario": uncertain.get("uncertain_scenario") or "unknown",
                "planned_event_count": uncertain.get("planned_event_count", 0) or 0,
                "executed_event_count": uncertain.get("executed_event_count", 0) or 0,
                "executed_event_types": ";".join(uncertain.get("executed_event_types") or []),
                "seed": clean.get("seed"),
                "model_sha256": clean.get("model_sha256"),
                "uncertain_event_file": uncertain.get("uncertain_event_file"),
                "successful_event_count": uncertain.get("successful_event_count"),
            })
            signature, category = event_signature(uncertain.get("executed_event_types"))
            rows[-1]["event_signature"] = signature
            rows[-1]["event_category"] = category
    return rows


def write_paired(rows, output):
    fields = ["clean_experiment_id", "uncertain_experiment_id", "agent", "family", "instance", "run",
              "clean_makespan", "uncertain_makespan", "delta_makespan", "relative_degradation",
              "uncertain_scenario", "planned_event_count", "executed_event_count", "executed_event_types",
              "event_signature", "event_category", "seed", "model_sha256", "uncertain_event_file",
              "successful_event_count"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if row["relative_degradation"] is not None:
                row["relative_degradation"] = f"{row['relative_degradation']:.6f}"
            writer.writerow(row)

    stats_path = output.with_name(output.stem + "_stats.csv")
    def write_stats(stat_rows, path, group_key_names):
        grouped = defaultdict(list)
        for row in stat_rows:
            grouped[tuple(row.get(name, "") for name in group_key_names)].append(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([*group_key_names, "n_pairs",
                             "clean_makespan_mean", "clean_makespan_std",
                             "clean_makespan_median",
                             "uncertain_makespan_mean", "uncertain_makespan_std",
                             "uncertain_makespan_median",
                             "delta_makespan_mean", "delta_makespan_std",
                             "delta_makespan_median",
                             "relative_degradation_mean", "relative_degradation_std",
                             "relative_degradation_median",
                             "positive_degradation_rate", "event_execution_rate",
                             "planned_event_count", "executed_event_count"])
            for key, items in sorted(grouped.items()):
                clean = [r["clean_makespan"] for r in items]
                uncertain = [r["uncertain_makespan"] for r in items]
                delta = [r["delta_makespan"] for r in items]
                relative = [r["relative_degradation"] for r in items if r["relative_degradation"] is not None]
                clean_mean, clean_std = mean_std(clean)
                uncertain_mean, uncertain_std = mean_std(uncertain)
                delta_mean, delta_std = mean_std(delta)
                relative_mean, relative_std = mean_std(relative)
                planned = sum(int(r["planned_event_count"]) for r in items)
                executed = sum(int(r["executed_event_count"]) for r in items)
                writer.writerow([*key, len(items), clean_mean, clean_std, median(clean),
                                 uncertain_mean, uncertain_std, median(uncertain),
                                 delta_mean, delta_std, median(delta),
                                 relative_mean, relative_std, median(relative),
                                 f"{sum(value > 0 for value in delta) / len(delta):.6f}",
                                 f"{executed / planned:.6f}" if planned else "",
                                 planned, executed])

    write_stats(rows, stats_path, ["agent", "family", "uncertain_scenario"])

    event_stats_path = output.with_name(output.stem + "_event_stats.csv")
    write_stats(rows, event_stats_path, ["agent", "family", "uncertain_scenario", "event_category", "event_signature"])

    # A trial containing several event types contributes to every type it
    # actually experienced.  This presence view answers questions such as
    # “包含 JOB_ADD 的试验平均退化多少”，while the signature view above keeps
    # exact mixed combinations separate.
    presence_rows = []
    for row in rows:
        signature = row.get("event_signature", "NONE")
        event_names = [name for name in signature.split("+") if name and name != "NONE"]
        if not event_names:
            event_names = ["NONE"]
        for name in event_names:
            copy_row = dict(row)
            copy_row["event_category"] = name
            presence_rows.append(copy_row)
    event_presence_stats_path = output.with_name(output.stem + "_event_presence_stats.csv")
    write_stats(presence_rows, event_presence_stats_path, ["agent", "family", "uncertain_scenario", "event_category"])
    return stats_path, event_stats_path, event_presence_stats_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-ids", nargs="+", help="需要合并的 benchmark 实验 ID")
    parser.add_argument("--output", required=True, help="汇总 CSV 路径")
    parser.add_argument("--clean-experiment-ids", nargs="+", help="配对模式 clean 实验 ID 列表")
    parser.add_argument("--uncertain-experiment-ids", nargs="+", help="配对模式 uncertain 实验 ID 列表")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if bool(args.clean_experiment_ids) != bool(args.uncertain_experiment_ids):
        parser.error("clean 和 uncertain 实验 ID 必须同时提供")
    if args.clean_experiment_ids:
        stats_path, event_stats_path, event_presence_stats_path = write_paired(
            paired_rows(args.clean_experiment_ids, args.uncertain_experiment_ids), output
        )
        print(f"paired_summary={output}")
        print(f"paired_stats={stats_path}")
        print(f"paired_event_stats={event_stats_path}")
        print(f"paired_event_presence_stats={event_presence_stats_path}")
        return
    if not args.experiment_ids:
        parser.error("需要 --experiment-ids 或配对模式参数")
    records = [record for experiment_id in args.experiment_ids for record in load_records(experiment_id)]
    write_makespan_summary(records, output)
    write_status(records, output.with_name(output.stem + "_status.csv"))
    print(f"makespan_summary={output}")


if __name__ == "__main__":
    main()
