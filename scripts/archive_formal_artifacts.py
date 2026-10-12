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
    "graphppo_ablation_seed101_v2_20261010",
    "graphppo_ablation_seed102_v2_20261010",
    "graphppo_ablation_seed103_v2_20261010",
    "strict_discount_regression_20261011",
    "graphgrpo_ablation_flat_strict2_seed101_v1_20261010",
    "graphgrpo_ablation_flat_strict2_seed102_v1_20261010",
    "graphgrpo_ablation_flat_strict2_seed103_v1_20261010",
    "graphgrpo_ablation_timeonly_strict2_seed101_v1_20261010",
    "graphgrpo_ablation_timeonly_strict2_seed102_v1_20261010",
    "graphgrpo_ablation_timeonly_strict2_seed103_v1_20261010",
    "graphgrpo_ablation_firstagv_strict2_seed101_v2_20261010",
    "graphgrpo_ablation_firstagv_strict2_seed102_v2_20261010",
    "graphgrpo_ablation_firstagv_strict2_seed103_v2_20261010",
    "graphgrpo_formal_final_seed101_v1_20261011",
    "graphgrpo_formal_final_seed102_v1_20261011",
    "graphgrpo_formal_final_seed103_v1_20261011",
    "hgs_formal_final_seed101_v1_20261011",
    "hgs_formal_final_seed102_v1_20261011",
    "hgs_formal_final_seed103_v1_20261011",
    *[
        f"graphgrpo_ablation_{variant}_final_seed{seed}_v1_20261012"
        for variant in ("fixedgnn", "flat", "timeonly", "firstagv")
        for seed in (101, 102, 103)
    ],
    *[
        f"graphgrpo_ablation_lb_{variant}_seed{seed}_v1_20261012"
        for variant in ("fixedgnn", "flat", "firstagv")
        for seed in (101, 102, 103)
    ],
    *[
        f"graphgrpo_ablation_timeonly_frozen_seed{seed}_v1_20261012"
        for seed in (101, 102, 103)
    ],
    *[f"graphgrpo_frozen_lb_seed{seed}_v1_20261012" for seed in (101, 102, 103)],
    "graphgrpo_frozen_lb_seed104_v1_20261012",
    "graphgrpo_frozen_lb_seed105_v1_20261012",
    *[f"graphgrpo_frozen_ms_seed{seed}_v1_20261012" for seed in (101, 102, 103)],
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
    *[
        f"graphppo_ablation_seed{seed}_test_{scene}_v2_20261010"
        for seed, scene in (
            (101, "base"), (101, "event"),
            (102, "base"), (102, "event"),
            (103, "base"), (103, "event"),
        )
    ],
    *[
        f"graphgrpo_formal_{scenario}_seed{seed}_v1_20261011"
        for scenario in ("machine_only", "agv_only", "job_add_only")
        for seed in (101, 102, 103)
    ],
    *[
        f"graphgrpo_formal_final_seed{seed}_test_{scene}_v1_20261011"
        for seed in (101, 102, 103)
        for scene in ("base", "event")
    ],
    *[
        f"hgs_formal_final_seed{seed}_test_{scene}_v1_20261011"
        for seed in (101, 102)
        for scene in ("base", "event")
    ],
    "hgs_formal_final_seed103_test_base_v1_20261011",
    "hgs_formal_final_seed103_test_event_v2_20261011",
    *[
        f"graphgrpo_formal_final_{scenario}_seed{seed}_v2_20261011"
        for scenario in ("machine_only", "agv_only", "job_add_only")
        for seed in (101, 102, 103)
    ],
    *[
        f"graphgrpo_ablation_{variant}_final_seed{seed}_test_{scene}_v1_20261012"
        for variant in ("fixedgnn", "flat", "timeonly", "firstagv")
        for seed in (101, 102, 103)
        for scene in ("base", "event")
    ],
    *[
        f"graphgrpo_ablation_lb_{variant}_seed{seed}_test_{scene}_v1_20261012"
        for variant in ("fixedgnn", "flat", "firstagv")
        for seed in (101, 102, 103)
        for scene in ("base", "event")
    ],
    *[
        f"graphgrpo_ablation_timeonly_frozen_seed{seed}_test_{scene}_v1_20261012"
        for seed in (101, 102, 103)
        for scene in ("base", "event")
    ],
    *[f"graphgrpo_frozen_lb_test_seed{seed}_{scene}_v2_20261012"
      for seed in (601, 602, 603) for scene in ("clean", "uncertain")],
    *[f"graphgrpo_frozen_ms_test_seed{seed}_{scene}_v1_20261012"
      for seed in (601, 602, 603) for scene in ("clean", "uncertain")],
    *[f"graphgrpo_frozen_lb_test_seed{seed}_{scenario}_v1_20261012"
      for seed in (601, 602, 603)
      for scenario in ("machine_only", "agv_only", "job_add_only")],
    *[
        f"graphgrpo_frozen_lb_test_model{model}_seed{test}_{scene}_v3_20261012"
        for model, test in ((101, 602), (101, 603))
        for scene in ("clean", "uncertain")
    ],
    *[
        f"graphgrpo_frozen_lb_test_m{model}_t{test}_{scene}_v3_20261012"
        for model, test in ((102, 601), (102, 603), (103, 601), (103, 602))
        for scene in ("clean", "event")
    ],
    *[
        f"graphgrpo_frozen_lb_test_seed{model}_{scene}_v2_20261012"
        for model in (104, 105)
        for scene in ("clean", "uncertain")
    ],
    *[
        f"hgs_formal_cross_m{model}_t{test}_{scene}_v1_20261012"
        for model, test in ((101, 602), (101, 603), (102, 601), (102, 603), (103, 601), (103, 602))
        for scene in ("base", "event")
    ],
    "ortools_formal_cross_t602_base_v1_20261012",
    "ortools_formal_cross_t602_event_v1_20261012",
    "ortools_formal_cross_t603_base_v1_20261012",
    "ortools_formal_cross_t603_event_v1_20261012",
]


EVENT_FILES = [
    "dataset/uncertain-events/default/brandimarte/mk11_agv_events.json",
    "dataset/uncertain-events/default/brandimarte/mk12_agv_events.json",
    "dataset/uncertain-events/default/brandimarte/mk13_agv_events.json",
    "dataset/uncertain-events/default/brandimarte/mk14_agv_events.json",
    "dataset/uncertain-events/default/brandimarte/mk15_agv_events.json",
    "dataset/uncertain-events/default/kacem/k4_agv_events.json",
]

#专项事件文件使用与正式测试集相同的 6 个基础实例。保留默认混合事件
#文件，同时归档仅机器故障、仅 AGV 故障和仅新作业到达三类时间线。
EVENT_FILES.extend(
    f"dataset/uncertain-events/{scenario}/{family}/{instance}_agv_events.json"
    for scenario in ("machine_only", "agv_only", "job_add_only")
    for family, instances in (("brandimarte", ("mk11", "mk12", "mk13", "mk14", "mk15")),
                              ("kacem", ("k4",)))
    for instance in instances
)

SUMMARY_FILES = [
    "benchmark_plots/formal_main_summary_v4.csv",
    "benchmark_plots/formal_main_instance_summary_v4.csv",
    "benchmark_plots/formal_main_status_summary_v4.csv",
    "benchmark_plots/formal_main_runtime_summary_v4.csv",
    "benchmark_plots/formal_main_comparison_v4.csv",
    "benchmark_plots/formal_main_paired_statistics_v4.csv",
    "benchmark_plots/formal_main_paired_comparisons_v4.csv",
    "benchmark_plots/formal_main_graphgrpo_final_v4_pair.csv",
    "benchmark_plots/formal_main_hgs_final_v4_pair.csv",
    "benchmark_plots/formal_main_ortools_v4_pair.csv",
    "benchmark_plots/formal_ablation_comparison_v1.csv",
    "benchmark_plots/formal_ablation_paired_statistics_v1.csv",
    "benchmark_plots/formal_ablation_paired_comparisons_v1.csv",
    "benchmark_plots/formal_graphgrpo_ablation_fixedgnn_final_seed101_103_pair_v1_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_flat_final_seed101_103_pair_v1_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_timeonly_final_seed101_103_pair_v1_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_firstagv_final_seed101_103_pair_v1_stats.csv",
    "benchmark_plots/formal_ablation_frozen_comparison_v1.csv",
    "benchmark_plots/formal_ablation_frozen_paired_statistics_v1.csv",
    "benchmark_plots/formal_ablation_frozen_paired_comparisons_v1.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_fixedgnn_seed601_603_pair_v1.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_fixedgnn_seed601_603_pair_v1_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_fixedgnn_seed601_603_pair_v1_event_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_fixedgnn_seed601_603_pair_v1_event_presence_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_flat_seed601_603_pair_v1.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_flat_seed601_603_pair_v1_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_flat_seed601_603_pair_v1_event_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_flat_seed601_603_pair_v1_event_presence_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_firstagv_seed601_603_pair_v1.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_firstagv_seed601_603_pair_v1_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_firstagv_seed601_603_pair_v1_event_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_lb_firstagv_seed601_603_pair_v1_event_presence_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_timeonly_frozen_seed601_603_pair_v1.csv",
    "benchmark_plots/formal_graphgrpo_ablation_timeonly_frozen_seed601_603_pair_v1_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_timeonly_frozen_seed601_603_pair_v1_event_stats.csv",
    "benchmark_plots/formal_graphgrpo_ablation_timeonly_frozen_seed601_603_pair_v1_event_presence_stats.csv",
    "benchmark_plots/formal_graphgrpo_final_machine_only_seed101_103_pair_v2_stats.csv",
    "benchmark_plots/formal_graphgrpo_final_agv_only_seed101_103_pair_v2_stats.csv",
    "benchmark_plots/formal_graphgrpo_final_job_add_only_seed101_103_pair_v2_stats.csv",
    "benchmark_plots/formal_main_frozen_comparison_v1.csv",
    "benchmark_plots/formal_main_frozen_instance_v1.csv",
    "benchmark_plots/formal_main_frozen_status_v1.csv",
    "benchmark_plots/formal_main_frozen_runtime_v1.csv",
    "benchmark_plots/formal_main_frozen_paired_comparison_v1.csv",
    "benchmark_plots/formal_main_frozen_event_comparison_v1.csv",
    "benchmark_plots/formal_main_frozen_event_presence_comparison_v1.csv",
    "benchmark_plots/formal_test_instance_metadata_v1.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_seed601_603_pair_v2.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_seed601_603_pair_v2_stats.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_seed601_603_pair_v2_event_stats.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_seed601_603_pair_v2_event_presence_stats.csv",
    "benchmark_plots/formal_graphgrpo_makespan_delta_seed601_603_pair_v1.csv",
    "benchmark_plots/formal_graphgrpo_makespan_delta_seed601_603_pair_v1_stats.csv",
    "benchmark_plots/formal_graphgrpo_makespan_delta_seed601_603_pair_v1_event_stats.csv",
    "benchmark_plots/formal_graphgrpo_makespan_delta_seed601_603_pair_v1_event_presence_stats.csv",
    *[f"benchmark_plots/formal_graphgrpo_frozen_{scenario}_seed601_603_pair_v1{suffix}.csv"
      for scenario in ("machine_only", "agv_only", "job_add_only")
      for suffix in ("", "_stats", "_event_stats", "_event_presence_stats")],
    "benchmark_plots/formal_graphgrpo_frozen_lb_five_seed_pairs_v1.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_five_seed_by_training_seed_v1.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_five_seed_by_instance_v1.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_five_seed_paired_statistics_v1.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_three_seed_pair_balanced_v2.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_three_seed_pair_balanced_v2_stats.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_three_seed_pair_balanced_v2_event_stats.csv",
    "benchmark_plots/formal_graphgrpo_frozen_lb_three_seed_pair_balanced_v2_event_presence_stats.csv",
    "benchmark_plots/formal_hgs_adapted_three_seed_pair_balanced_v2.csv",
    "benchmark_plots/formal_hgs_adapted_three_seed_pair_balanced_v2_stats.csv",
    "benchmark_plots/formal_hgs_adapted_three_seed_pair_balanced_v2_event_stats.csv",
    "benchmark_plots/formal_hgs_adapted_three_seed_pair_balanced_v2_event_presence_stats.csv",
    "benchmark_plots/formal_ortools_three_seed_pair_balanced_v2.csv",
    "benchmark_plots/formal_ortools_three_seed_pair_balanced_v2_stats.csv",
    "benchmark_plots/formal_ortools_three_seed_pair_balanced_v2_event_stats.csv",
    "benchmark_plots/formal_ortools_three_seed_pair_balanced_v2_event_presence_stats.csv",
    "benchmark_plots/formal_main_frozen_balanced_comparison_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_instance_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_status_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_runtime_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_paired_comparison_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_event_comparison_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_event_presence_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_paired_statistics_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_test_conditions_v2.csv",
    "benchmark_plots/formal_main_frozen_balanced_paired_differences_v2.csv",
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

    summaries = []
    for relative in SUMMARY_FILES:
        path = ROOT / relative
        summaries.append({"path": relative, "sha256": sha256(path) if path.exists() else None})

    source_files = [
        "config/thesis_formal_split_v1.json",
        "scripts/auto_train_loop.py",
        "scripts/run_benchmark.py",
        "scripts/summarize_formal_results.py",
        "scripts/compare_formal_methods.py",
        "scripts/analyze_paired_statistics.py",
        "scripts/build_formal_summary.py",
        "scripts/build_balanced_graphgrpo_summary.py",
        "scripts/analyze_balanced_main_methods.py",
        "scripts/check_thesis_final.py",
        "scripts/build_thesis_integrated.py",
        "scripts/summarize_test_instance_metadata.py",
        "scripts/build_thesis_integrated.py",
        "executor/packet_factory/call_back/callback_manager/CallbackManager.py",
        "executor/packet_factory/lifecycle/initializer/env_initializer.py",
        "executor/packet_factory/packet_factory/packet_factory_env/packet_factory_env.py",
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
        "result_validity_notes": [
            "The final main comparison uses the five-round checkpoints and the corrected HGS seed 103 event benchmark.",
        "The balanced main comparison evaluates the three learning checkpoints and OR-Tools under test seeds 601--603; training seeds 104 and 105 are retained as a separate stability audit for the proposed method.",
            "Strict-group v3 benchmark results were generated before decision-batch discounting was corrected and remain historical; strict-group v4 is retained as diagnostic evidence.",
            "strict_discount_regression_20261011 verifies the corrected strict-group return computation on a two-trajectory smoke run.",
        ],
        "training_experiments": training,
        "benchmark_experiments": benchmarks,
        "event_files": events,
        "summary_files": summaries,
        "source_and_dependency_hashes": sources,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
