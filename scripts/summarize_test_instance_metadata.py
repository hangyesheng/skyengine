#!/usr/bin/env python3
"""Summarize the structural and event-load metadata of formal test instances.

The summary is an evidence artifact for the thesis.  It is derived from the
original AGV instance files and the frozen default event timelines; benchmark
results are not used to infer instance size or event workload.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCES = {
    "brandimarte": ["mk11", "mk12", "mk13", "mk14", "mk15"],
    "kacem": ["k4"],
}


def parse_instance(path: Path) -> dict:
    lines = [line.strip().split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs, machines, agvs, points, links = map(int, lines[0][:5])
    cursor = 1
    initial_operations = 0
    for _ in range(jobs):
        row = list(map(int, lines[cursor]))
        cursor += 1
        n_ops = row[0]
        initial_operations += n_ops
        pos = 1
        for _ in range(n_ops):
            choices = row[pos]
            pos += 1 + 2 * choices
    cursor += points  # point records
    link_weights = []
    for _ in range(links):
        row = list(map(float, lines[cursor]))
        cursor += 1
        link_weights.append(row[3])
    cursor += machines  # machine-to-point records
    velocities = []
    for _ in range(agvs):
        row = list(map(float, lines[cursor]))
        cursor += 1
        velocities.append(row[2])
    return {
        "jobs": jobs,
        "machines": machines,
        "agvs": agvs,
        "points": points,
        "links": links,
        "initial_operations": initial_operations,
        "agv_velocity_min": min(velocities) if velocities else 0.0,
        "agv_velocity_max": max(velocities) if velocities else 0.0,
        "agv_velocity_mean": mean(velocities) if velocities else 0.0,
        "link_weight_min": min(link_weights) if link_weights else 0.0,
        "link_weight_max": max(link_weights) if link_weights else 0.0,
        "link_weight_mean": mean(link_weights) if link_weights else 0.0,
        # The converter used for all formal runs assigns this common capacity.
        "agv_capacity": 12,
    }


def event_metadata(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = data.get("event_counts", {})
    added_jobs = 0
    added_operations = 0
    event_record_counts = {"machine_fail": 0, "agv_fail": 0, "job_add": 0}
    trigger_counts = {"machine_fail": 0, "agv_fail": 0, "job_add": 0}
    for event in data.get("event_timeline", []):
        event_type = str(event.get("type", "")).split(".")[-1]
        args = event.get("args", [])
        phase = args[0] if args and isinstance(args[0], str) else None
        if event_type == "JOB_ADD":
            event_record_counts["job_add"] += 1
            payload = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
            job = payload.get("job", {}) if isinstance(payload, dict) else {}
            if phase == "trigger":
                trigger_counts["job_add"] += 1
                added_jobs += 1
                added_operations += len(job.get("operations", []))
        elif event_type == "MACHINE_FAIL":
            event_record_counts["machine_fail"] += 1
            if phase == "trigger":
                trigger_counts["machine_fail"] += 1
        elif event_type == "AGV_FAIL":
            event_record_counts["agv_fail"] += 1
            if phase == "trigger":
                trigger_counts["agv_fail"] += 1
    return {
        "planned_event_records": len(data.get("event_timeline", [])),
        "planned_machine_fail_records": event_record_counts["machine_fail"],
        "planned_agv_fail_records": event_record_counts["agv_fail"],
        "planned_machine_failures": trigger_counts["machine_fail"],
        "planned_agv_failures": trigger_counts["agv_fail"],
        "planned_job_additions": trigger_counts["job_add"],
        "added_jobs": added_jobs,
        "added_operations": added_operations,
        "event_seed": data.get("generation_config", {}).get("seed"),
    }


def build_rows(root: Path) -> list[dict]:
    rows = []
    for family, instances in DEFAULT_INSTANCES.items():
        for instance in instances:
            instance_path = root / "dataset" / "agv-instances" / family / f"{instance}_agv.txt"
            event_path = root / "dataset" / "uncertain-events" / "default" / family / f"{instance}_agv_events.json"
            row = {"family": family, "instance": instance, "split": "test"}
            row.update(parse_instance(instance_path))
            row.update(event_metadata(event_path))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_plots" / "formal_test_instance_metadata_v1.csv")
    args = parser.parse_args()
    rows = build_rows(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
