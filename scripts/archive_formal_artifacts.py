#!/usr/bin/env python3
"""Write a compact manifest for the frozen formal-test evidence.

The manifest records experiment directories, checkpoint hashes, benchmark
result files, event-file hashes, and source/dependency hashes.  It is intended
for archival and review; it does not copy large checkpoints into the Git
repository.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


TRAINING_DIRS = [
    *[
        f"graphgrpo_ablation_{variant}_strict2_seed{seed}_v3_20261010"
        for variant in ("fixedgnn", "flat", "timeonly", "firstagv")
        for seed in (101, 102, 103)
    ],
    "graphgrpo_formal_strict2_seed101_v3_20261010",
    "graphgrpo_formal_strict2_seed102_v3_20261010",
    "graphgrpo_formal_strict2_seed103_v3_20261010",
    "graphgrpo_formal_default_seed101_v3_20261010",
    "graphgrpo_formal_default_seed102_v3_20261010",
    "graphgrpo_formal_default_seed103_v3_20261010",
    "graphgrpo_formal_default_seed101_v2_20261010",
    "graphgrpo_formal_default_seed102_v1_20261010",
    "graphgrpo_formal_default_seed103_v1_20261010",
    "graphgrpo_formal_strict2_seed101_v2_20261010",
    "graphgrpo_formal_strict2_seed102_v1_20261010",
    "graphgrpo_formal_strict2_seed103_v1_20261010",
    "hgs_formal_seed101_v1_20261010",
    "hgs_formal_seed102_v1_20261010",
    "hgs_formal_seed103_v1_20261010",
    "graphgrpo_ablation_fixedgnn_strict2_seed101_v1_20261010",
    "graphgrpo_ablation_fixedgnn_strict2_seed102_v1_20261010",
    "graphgrpo_ablation_fixedgnn_strict2_seed103_v1_20261010",
    "graphppo_ablation_seed101_v1_20261010",
    "graphppo_ablation_seed102_v1_20261010",
    "graphppo_ablation_seed103_v1_20261010",
    "graphgrpo_ablation_flat_strict2_seed101_v1_20261010",
    "graphgrpo_ablation_flat_strict2_seed102_v1_20261010",
    "graphgrpo_ablation_flat_strict2_seed103_v1_20261010",
    "graphgrpo_ablation_timeonly_strict2_seed101_v1_20261010",
    "graphgrpo_ablation_timeonly_strict2_seed102_v1_20261010",
    "graphgrpo_ablation_timeonly_strict2_seed103_v1_20261010",
    "graphgrpo_ablation_firstagv_strict2_seed101_v2_20261010",
    "graphgrpo_ablation_firstagv_strict2_seed102_v2_20261010",
    "graphgrpo_ablation_firstagv_strict2_seed103_v2_20261010",
]


BENCHMARK_DIRS = [
    *[
        f"graphgrpo_ablation_{variant}_seed{seed}_test_{scene}_v3_20261010"
        for variant in ("fixedgnn", "flat", "timeonly", "firstagv")
        for seed in (101, 102, 103)
        for scene in ("clean", "uncertain")
    ],
    "graphgrpo_formal_strict2_seed101_test_clean_v3_20261010",
    "graphgrpo_formal_strict2_seed101_test_uncertain_v3_20261010",
    "graphgrpo_formal_strict2_seed102_test_clean_v3_20261010",
    "graphgrpo_formal_strict2_seed102_test_uncertain_v3_20261010",
    "graphgrpo_formal_strict2_seed103_test_clean_v3_20261010",
    "graphgrpo_formal_strict2_seed103_test_uncertain_v3_20261010",
    "graphgrpo_formal_default_seed101_test_clean_v3_20261010",
    "graphgrpo_formal_default_seed101_test_uncertain_v3_20261010",
    "graphgrpo_formal_default_seed102_test_clean_v3_20261010",
    "graphgrpo_formal_default_seed102_test_uncertain_v3_20261010",
    "graphgrpo_formal_default_seed103_test_clean_v3_20261010",
    "graphgrpo_formal_default_seed103_test_uncertain_v3_20261010",
    "graphgrpo_formal_default_seed101_test_clean_v2_20261010",
    "graphgrpo_formal_default_seed101_test_uncertain_v2_20261010",
    "graphgrpo_formal_default_seed102_test_clean_v2_20261010",
    "graphgrpo_formal_default_seed102_test_uncertain_v2_20261010",
    "graphgrpo_formal_default_seed103_test_clean_v2_20261010",
    "graphgrpo_formal_default_seed103_test_uncertain_v2_20261010",
    "graphgrpo_formal_strict2_seed101_test_clean_v2_20261010",
    "graphgrpo_formal_strict2_seed101_test_uncertain_v2_20261010",
    "graphgrpo_formal_strict2_seed102_test_clean_v1_20261010",
    "graphgrpo_formal_strict2_seed102_test_uncertain_v1_20261010",
    "graphgrpo_formal_strict2_seed103_test_clean_v1_20261010",
    "graphgrpo_formal_strict2_seed103_test_uncertain_v1_20261010",
    "hgs_formal_seed101_test_clean_v1_20261010",
    "hgs_formal_seed101_test_uncertain_v1_20261010",
    "hgs_formal_seed102_test_clean_v1_20261010",
    "hgs_formal_seed102_test_uncertain_v1_20261010",
    "hgs_formal_seed103_test_clean_v1_20261010",
    "hgs_formal_seed103_test_uncertain_v1_20261010",
    "graphgrpo_ablation_fixedgnn_seed101_test_clean_v1_20261010",
    "graphgrpo_ablation_fixedgnn_seed101_test_uncertain_v1_20261010",
    "graphgrpo_ablation_fixedgnn_seed102_test_clean_v1_20261010",
    "graphgrpo_ablation_fixedgnn_seed102_test_uncertain_v1_20261010",
    "graphgrpo_ablation_fixedgnn_seed103_test_clean_v1_20261010",
    "graphgrpo_ablation_fixedgnn_seed103_test_uncertain_v1_20261010",
    "graphppo_ablation_seed101_test_clean_v1_20261010",
    "graphppo_ablation_seed101_test_uncertain_v1_20261010",
    "graphppo_ablation_seed102_test_clean_v1_20261010",
    "graphppo_ablation_seed102_test_uncertain_v1_20261010",
    "graphppo_ablation_seed103_test_clean_v1_20261010",
    "graphppo_ablation_seed103_test_uncertain_v1_20261010",
    "graphgrpo_ablation_flat_seed101_test_clean_v1_20261010",
    "graphgrpo_ablation_flat_seed101_test_uncertain_v1_20261010",
    "graphgrpo_ablation_flat_seed102_test_clean_v1_20261010",
    "graphgrpo_ablation_flat_seed102_test_uncertain_v1_20261010",
    "graphgrpo_ablation_flat_seed103_test_clean_v1_20261010",
    "graphgrpo_ablation_flat_seed103_test_uncertain_v1_20261010",
    "graphgrpo_ablation_timeonly_seed101_test_clean_v1_20261010",
    "graphgrpo_ablation_timeonly_seed101_test_uncertain_v1_20261010",
    "graphgrpo_ablation_timeonly_seed102_test_clean_v1_20261010",
    "graphgrpo_ablation_timeonly_seed102_test_uncertain_v1_20261010",
    "graphgrpo_ablation_timeonly_seed103_test_clean_v1_20261010",
    "graphgrpo_ablation_timeonly_seed103_test_uncertain_v1_20261010",
    "graphgrpo_ablation_firstagv_seed101_test_clean_v2_20261010",
    "graphgrpo_ablation_firstagv_seed101_test_uncertain_v2_20261010",
    "graphgrpo_ablation_firstagv_seed102_test_clean_v2_20261010",
    "graphgrpo_ablation_firstagv_seed102_test_uncertain_v2_20261010",
    "graphgrpo_ablation_firstagv_seed103_test_clean_v2_20261010",
    "graphgrpo_ablation_firstagv_seed103_test_uncertain_v2_20261010",
    "ortools_formal_seed601_test_clean_v1_20261010",
    "ortools_formal_seed601_test_uncertain_v1_20261010",
]


EVENT_FILES = [
    "dataset/uncertain-events/default/brandimarte/mk11_agv_events.json",
    "dataset/uncertain-events/default/brandimarte/mk12_agv_events.json",
    "dataset/uncertain-events/default/brandimarte/mk13_agv_events.json",
    "dataset/uncertain-events/default/brandimarte/mk14_agv_events.json",
    "dataset/uncertain-events/default/brandimarte/mk15_agv_events.json",
    "dataset/uncertain-events/default/kacem/k4_agv_events.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_status():
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
        return {
            "dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"dirty": None, "status_sha256": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/thesis/formal_artifact_manifest_v1.json")
    args = parser.parse_args()

    training = []
    for experiment_id in TRAINING_DIRS:
        directory = ROOT / "training_logs" / "results" / experiment_id
        config = directory / "experiment_config.json"
        checkpoint = directory / "checkpoints" / "best.pt"
        meta = directory / "checkpoints" / "best_checkpoint_meta.json"
        training.append({
            "experiment_id": experiment_id,
            "experiment_config": str(config.relative_to(ROOT)) if config.exists() else None,
            "best_checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.exists() else None,
            "best_checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else None,
            "best_checkpoint_meta": json.loads(meta.read_text()) if meta.exists() else None,
        })

    benchmarks = []
    for experiment_id in BENCHMARK_DIRS:
        directory = ROOT / "benchmark_results" / experiment_id
        config = directory / "experiment_config.json"
        results = directory / "results.jsonl"
        benchmarks.append({
            "experiment_id": experiment_id,
            "experiment_config": str(config.relative_to(ROOT)) if config.exists() else None,
            "results": str(results.relative_to(ROOT)) if results.exists() else None,
            "results_sha256": sha256(results) if results.exists() else None,
        })

    events = []
    for relative in EVENT_FILES:
        path = ROOT / relative
        events.append({"path": relative, "sha256": sha256(path) if path.exists() else None})

    source_files = [
        "config/thesis_formal_split_v1.json",
        "scripts/auto_train_loop.py",
        "scripts/run_benchmark.py",
        "scripts/summarize_formal_results.py",
        "scripts/compare_formal_methods.py",
        "scripts/analyze_paired_statistics.py",
        "executor/packet_factory/packet_factory/Agent/GraphGRPOAgent.py",
        "pyproject.toml",
        "uv.lock",
    ]
    sources = [{"path": path, "sha256": sha256(ROOT / path) if (ROOT / path).exists() else None} for path in source_files]

    manifest = {
        "manifest_version": "formal_artifact_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "git_worktree": git_worktree_status(),
        "protocol": "config/thesis_formal_split_v1.json",
        "training_experiments": training,
        "benchmark_experiments": benchmarks,
        "event_files": events,
        "source_and_dependency_hashes": sources,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
