#!/usr/bin/env python3
"""Combine paired clean/uncertain summaries from several formal methods.

Each input is a paired CSV produced by ``summarize_formal_results.py``.  The
script keeps the method label supplied on the command line, then computes one
comparison table by dataset family and a second table by executed event
category.  It deliberately aggregates paired rows rather than already
aggregated means, so every test instance/seed remains one observation.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def mean_std(values):
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return f"{mean:.6f}", f"{math.sqrt(variance):.6f}"


def read_rows(spec):
    if "=" not in spec:
        raise ValueError(f"方法参数必须是 LABEL=CSV_PATH: {spec}")
    label, path = spec.split("=", 1)
    if not label or not path:
        raise ValueError(f"方法参数不能为空: {spec}")
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["method"] = label
        for field in ("clean_makespan", "uncertain_makespan", "delta_makespan",
                      "relative_degradation", "planned_event_count", "executed_event_count"):
            row[field] = float(row[field]) if row.get(field) not in (None, "") else 0.0
        if (not math.isfinite(row["clean_makespan"]) or row["clean_makespan"] <= 0
                or not math.isfinite(row["uncertain_makespan"])):
            raise ValueError(f"无效的 makespan: {label} / {row.get('instance')}")
        # Recompute from the unrounded makespans. The exported paired CSV
        # rounds relative degradation for display, so averaging that column
        # can otherwise disagree with the original JSONL aggregation.
        row["delta_makespan"] = row["uncertain_makespan"] - row["clean_makespan"]
        row["relative_degradation"] = row["delta_makespan"] / row["clean_makespan"]
        row.setdefault("event_category", "MIXED")
        row.setdefault("event_signature", row.get("executed_event_types", ""))
    return rows


def write_grouped(rows, output, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    fields = [*keys, "n_pairs", "clean_makespan_mean", "clean_makespan_std",
              "uncertain_makespan_mean", "uncertain_makespan_std",
              "delta_makespan_mean", "delta_makespan_std",
              "relative_degradation_mean", "relative_degradation_std",
              "positive_degradation_rate", "event_execution_rate",
              "planned_event_count", "executed_event_count"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for key, items in sorted(grouped.items()):
            clean = [r["clean_makespan"] for r in items]
            uncertain = [r["uncertain_makespan"] for r in items]
            delta = [r["delta_makespan"] for r in items]
            relative = [r["relative_degradation"] for r in items]
            clean_mean, clean_std = mean_std(clean)
            uncertain_mean, uncertain_std = mean_std(uncertain)
            delta_mean, delta_std = mean_std(delta)
            relative_mean, relative_std = mean_std(relative)
            planned = sum(int(r["planned_event_count"]) for r in items)
            executed = sum(int(r["executed_event_count"]) for r in items)
            writer.writerow([
                *key, len(items), clean_mean, clean_std, uncertain_mean, uncertain_std,
                delta_mean, delta_std, relative_mean, relative_std,
                f"{sum(value > 0 for value in delta) / len(delta):.6f}",
                f"{executed / planned:.6f}" if planned else "", planned, executed,
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True,
                        help="方法与配对 CSV，格式 LABEL=path.csv；可重复传入")
    parser.add_argument("--output", required=True, help="按方法/数据族汇总的 CSV")
    parser.add_argument("--event-output", help="按方法/数据族/事件类型汇总的 CSV")
    parser.add_argument("--event-presence-output", help="按事件是否出现汇总的 CSV")
    args = parser.parse_args()

    rows = [row for spec in args.method for row in read_rows(spec)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_grouped(rows, output, ["method", "family"])
    event_output = Path(args.event_output) if args.event_output else output.with_name(output.stem + "_event_stats.csv")
    write_grouped(rows, event_output, ["method", "family", "event_category", "event_signature"])
    presence_rows = []
    for row in rows:
        names = [name for name in row.get("event_signature", "NONE").split("+") if name and name != "NONE"]
        if not names:
            names = ["NONE"]
        for name in names:
            copy_row = dict(row)
            copy_row["event_category"] = name
            presence_rows.append(copy_row)
    event_presence_output = (Path(args.event_presence_output) if args.event_presence_output
                             else output.with_name(output.stem + "_event_presence_stats.csv"))
    write_grouped(presence_rows, event_presence_output, ["method", "family", "event_category"])
    print(f"comparison={output}")
    print(f"event_comparison={event_output}")
    print(f"event_presence_comparison={event_presence_output}")


if __name__ == "__main__":
    main()
