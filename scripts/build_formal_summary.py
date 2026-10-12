#!/usr/bin/env python3
"""Build one auditable summary for the frozen formal benchmark batches.

The existing paired CSVs are useful for degradation statistics, but they only
contain successful pairs.  This report also counts every planned trial and
keeps completion rates and decision cost beside the paired makespan metrics.
Each ``--method`` argument has the form::

    LABEL=CLEAN_ID,EVENT_ID[,CLEAN_ID,EVENT_ID...]

The clean/event IDs are paired in order.  The script reads the raw JSONL
records and the experiment configurations, then writes three CSV files:

* the method/family summary used for the main table;
* an instance-level audit table;
* status and runtime summaries.

Only completed runs with a finite makespan enter the paired degradation
statistics.  Completion rates use all records, so failures cannot disappear
through paired filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from summarize_formal_results import load_records, mean_std, paired_rows


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"


def parse_method(spec: str):
    if "=" not in spec:
        raise ValueError(f"方法参数必须是 LABEL=CLEAN_ID,EVENT_ID: {spec}")
    label, value = spec.split("=", 1)
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not label or not tokens or len(tokens) % 2:
        raise ValueError(f"{spec}: clean/event 实验 ID 必须成对出现")
    return label, list(zip(tokens[::2], tokens[1::2]))


def config_seed(experiment_id: str):
    path = RESULTS_ROOT / experiment_id / "experiment_config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("seed")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def status_key(record):
    return (record.get("family", ""), record.get("instance", ""), record.get("run", ""))


def completed(record):
    value = record.get("makespan")
    return record.get("status") == "completed" and value is not None and math.isfinite(float(value))


def numeric_stats(records, field):
    values = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    mean, std = mean_std(values)
    return len(values), mean, std


def decision_value(record, field):
    stats = record.get("decision_stats") or {}
    value = stats.get(field)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(method_specs):
    main_rows = []
    instance_rows = []
    status_rows = []
    runtime_rows = []

    for label, pairs in method_specs:
        all_status = []
        all_runtime = []
        all_pair_rows = []
        for clean_id, event_id in pairs:
            clean_records = load_records(clean_id)
            event_records = load_records(event_id)
            clean_seed = config_seed(clean_id)
            event_seed = config_seed(event_id)
            if clean_seed != event_seed:
                raise ValueError(f"{clean_id}/{event_id}: benchmark seed 不一致")
            all_status.append(("基础场景", clean_records))
            all_status.append(("动态事件场景", event_records))
            all_runtime.append(("基础场景", clean_records))
            all_runtime.append(("动态事件场景", event_records))
            all_pair_rows.extend(paired_rows([clean_id], [event_id]))

        families = sorted({record.get("family", "") for _, records in all_status for record in records})
        for family in families:
            clean = [record for scenario, records in all_status if scenario == "基础场景" for record in records if record.get("family") == family]
            event = [record for scenario, records in all_status if scenario == "动态事件场景" for record in records if record.get("family") == family]
            pairs_for_family = [row for row in all_pair_rows if row.get("family") == family]
            clean_ok = [record for record in clean if completed(record)]
            event_ok = [record for record in event if completed(record)]
            clean_values = [float(record["makespan"]) for record in clean_ok]
            event_values = [float(record["makespan"]) for record in event_ok]
            delta_values = [float(row["delta_makespan"]) for row in pairs_for_family]
            relative_values = [float(row["relative_degradation"]) for row in pairs_for_family]
            planned = sum(int(row.get("planned_event_count") or 0) for row in pairs_for_family)
            executed = sum(int(row.get("executed_event_count") or 0) for row in pairs_for_family)
            clean_mean, clean_std = mean_std(clean_values)
            event_mean, event_std = mean_std(event_values)
            delta_mean, delta_std = mean_std(delta_values)
            relative_mean, relative_std = mean_std(relative_values)
            main_rows.append({
                "method": label,
                "family": family,
                "benchmark_batches": len(pairs),
                "benchmark_seeds": ";".join(str(config_seed(clean_id)) for clean_id, _ in pairs),
                "base_trials": len(clean),
                "base_completed": len(clean_ok),
                "base_completion_rate": f"{len(clean_ok) / len(clean):.6f}" if clean else "",
                "event_trials": len(event),
                "event_completed": len(event_ok),
                "event_completion_rate": f"{len(event_ok) / len(event):.6f}" if event else "",
                "paired_trials": len(pairs_for_family),
                "base_makespan_mean": clean_mean,
                "base_makespan_std": clean_std,
                "event_makespan_mean": event_mean,
                "event_makespan_std": event_std,
                "delta_makespan_mean": delta_mean,
                "delta_makespan_std": delta_std,
                "relative_degradation_mean": relative_mean,
                "relative_degradation_std": relative_std,
                "positive_degradation_rate": f"{sum(value > 0 for value in delta_values) / len(delta_values):.6f}" if delta_values else "",
                "planned_event_count": planned,
                "executed_event_count": executed,
                "event_execution_rate": f"{executed / planned:.6f}" if planned else "",
            })

            for instance in sorted({record.get("instance", "") for record in clean + event}):
                clean_i = [record for record in clean_ok if record.get("instance") == instance]
                event_i = [record for record in event_ok if record.get("instance") == instance]
                pair_i = [row for row in pairs_for_family if row.get("instance") == instance]
                cm, cs = mean_std([float(record["makespan"]) for record in clean_i])
                em, es = mean_std([float(record["makespan"]) for record in event_i])
                dm, ds = mean_std([float(row["delta_makespan"]) for row in pair_i])
                instance_rows.append({
                    "method": label, "family": family, "instance": instance,
                    "base_trials": sum(record.get("instance") == instance for record in clean),
                    "base_completed": len(clean_i),
                    "event_trials": sum(record.get("instance") == instance for record in event),
                    "event_completed": len(event_i), "paired_trials": len(pair_i),
                    "base_makespan_mean": cm, "base_makespan_std": cs,
                    "event_makespan_mean": em, "event_makespan_std": es,
                    "delta_makespan_mean": dm, "delta_makespan_std": ds,
                })

        status_counts = Counter((scenario, record.get("family", ""), record.get("status", "unknown"))
                                for scenario, records in all_status for record in records)
        for (scenario, family, status), count in sorted(status_counts.items()):
            status_rows.append({"method": label, "scenario": scenario, "family": family,
                                "status": status, "n_records": count})

        runtime_by_key = defaultdict(list)
        for scenario, records in all_runtime:
            for record in records:
                if completed(record):
                    runtime_by_key[(scenario, record.get("family", ""))].append(record)
        for (scenario, family), family_records in sorted(runtime_by_key.items()):
            for metric, field in (("decision_time_mean", "average_decision_time"),
                                  ("total_decision_time_mean", "total_decision_time"),
                                  ("elapsed_mean", "elapsed"),
                                  ("steps_mean", "steps")):
                values = []
                for record in family_records:
                    value = decision_value(record, field) if field in ("average_decision_time", "total_decision_time") else record.get(field)
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        values.append(value)
                mean, std = mean_std(values)
                runtime_rows.append({"method": label, "scenario": scenario, "family": family,
                                     "metric": metric, "n_completed": len(values),
                                     "mean": mean, "std": std})
    return main_rows, instance_rows, status_rows, runtime_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True,
                        help="方法参数：LABEL=CLEAN_ID,EVENT_ID[,CLEAN_ID,EVENT_ID...]")
    parser.add_argument("--output", required=True, help="方法/数据族汇总 CSV")
    parser.add_argument("--instance-output", required=True, help="实例级审计 CSV")
    parser.add_argument("--status-output", required=True, help="状态统计 CSV")
    parser.add_argument("--runtime-output", required=True, help="决策时间和运行时间 CSV")
    args = parser.parse_args()
    specs = [parse_method(value) for value in args.method]
    labels = [label for label, _ in specs]
    if len(labels) != len(set(labels)):
        parser.error("方法标签不能重复")
    main_rows, instance_rows, status_rows, runtime_rows = build(specs)
    write_csv(Path(args.output), list(main_rows[0]) if main_rows else [], main_rows)
    write_csv(Path(args.instance_output), list(instance_rows[0]) if instance_rows else [], instance_rows)
    write_csv(Path(args.status_output), list(status_rows[0]) if status_rows else [], status_rows)
    write_csv(Path(args.runtime_output), list(runtime_rows[0]) if runtime_rows else [], runtime_rows)
    print(f"summary={args.output}")
    print(f"instance_summary={args.instance_output}")
    print(f"status_summary={args.status_output}")
    print(f"runtime_summary={args.runtime_output}")


if __name__ == "__main__":
    main()
