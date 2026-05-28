"""
Benchmark 实验结果绘图脚本

功能：
1. 读取 results.jsonl 生成各数据集族的箱型图
2. 生成综合对比图（所有族在同一图中）
3. 输出汇总 CSV 表

用法：
    uv run python scripts/plot_benchmark.py --experiment-id bench_20260519_120000
    uv run python scripts/plot_benchmark.py --experiment-id my_exp --metric makespan
    uv run python scripts/plot_benchmark.py --experiment-id my_exp --metric avg_decision_time
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark_plots"

logger = logging.getLogger("benchmark_plot")

# Agent 颜色与显示名
AGENT_COLORS = {
    "DualDRLAgent": "#2196F3",
    "ORToolsAgent": "#4CAF50",
    "ORToolsBatchAgent": "#FF9800",
    "GraphDPAgent": "#9C27B0",
}

AGENT_LABELS = {
    "DualDRLAgent": "DualDRL",
    "ORToolsAgent": "OR-Tools",
    "ORToolsBatchAgent": "OR-Tools Batch",
    "GraphDPAgent": "GraphDP",
}

METRIC_LABELS = {
    "makespan": "Makespan",
    "avg_decision_time": "Avg Decision Time (s)",
    "total_decision_time": "Total Decision Time (s)",
    "decision_count": "Decision Count",
}


def setup_logging(log_level: str = "INFO"):
    level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    level = level_map.get(log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(level)


def load_results(results_path: Path) -> List[dict]:
    """读取 JSONL 结果文件"""
    records = []
    if not results_path.exists():
        logger.error(f"结果文件不存在: {results_path}")
        return records
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_metric(record: dict, metric: str) -> Optional[float]:
    """从记录中提取指标值"""
    if metric == "makespan":
        return record.get("makespan")
    elif metric == "avg_decision_time":
        stats = record.get("decision_stats", {})
        if isinstance(stats, dict):
            return stats.get("average_decision_time")
    elif metric == "total_decision_time":
        stats = record.get("decision_stats", {})
        if isinstance(stats, dict):
            return stats.get("total_decision_time")
    elif metric == "decision_count":
        stats = record.get("decision_stats", {})
        if isinstance(stats, dict):
            return stats.get("decision_count")
    return None


def group_data(records: List[dict], metric: str) -> Dict[str, Dict[str, List[float]]]:
    """按 family → agent 分组提取指标值

    Returns:
        {family: {agent: [values]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("status") != "completed":
            continue
        val = extract_metric(r, metric)
        if val is not None:
            family = r["family"]
            agent = r["agent"]
            grouped[family][agent].append(val)
    return dict(grouped)


def plot_family_boxplot(family: str, data: Dict[str, List[float]], metric: str, agents: List[str], output_dir: Path):
    """为单个族绘制箱型图"""
    fig, ax = plt.subplots(figsize=(8, 5))

    agent_data = []
    labels = []
    colors = []
    for agent in agents:
        if agent in data and data[agent]:
            agent_data.append(data[agent])
            labels.append(AGENT_LABELS.get(agent, agent))
            colors.append(AGENT_COLORS.get(agent, "#999999"))
        else:
            agent_data.append([])
            labels.append(AGENT_LABELS.get(agent, agent))
            colors.append("#DDDDDD")

    # 过滤掉空数据
    valid_indices = [i for i, d in enumerate(agent_data) if len(d) > 0]
    if not valid_indices:
        logger.warning(f"  {family}: 无有效数据，跳过")
        plt.close(fig)
        return

    valid_data = [agent_data[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    valid_colors = [colors[i] for i in valid_indices]

    bp = ax.boxplot(valid_data, patch_artist=True, widths=0.5, showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5})

    for patch, color in zip(bp["boxes"], valid_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(valid_labels, fontsize=11)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
    ax.set_title(f"{family}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 标注样本数
    for i, d in enumerate(valid_data):
        ax.text(i + 1, ax.get_ylim()[0], f"n={len(d)}", ha="center", va="bottom", fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(output_dir / f"{family}_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  保存: {family}_{metric}.png")


def plot_combined_boxplot(grouped: Dict[str, Dict[str, List[float]]], metric: str, agents: List[str], output_dir: Path):
    """绘制综合对比图：所有族在同一图中（2x4 子图网格）"""
    families = sorted(grouped.keys())
    n_families = len(families)
    if n_families == 0:
        logger.warning("无数据可绘制综合图")
        return

    ncols = min(4, n_families)
    nrows = (n_families + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, family in enumerate(families):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        data = grouped[family]
        agent_data = []
        labels = []
        colors = []

        for agent in agents:
            if agent in data and data[agent]:
                agent_data.append(data[agent])
                labels.append(AGENT_LABELS.get(agent, agent))
                colors.append(AGENT_COLORS.get(agent, "#999999"))

        if not agent_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(family, fontsize=11, fontweight="bold")
            continue

        bp = ax.boxplot(agent_data, patch_artist=True, widths=0.5, showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4})

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_title(family, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # 隐藏空子图
    for idx in range(n_families, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Benchmark: {METRIC_LABELS.get(metric, metric)}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"all_families_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"保存综合图: all_families_{metric}.png")


def generate_summary_csv(records: List[dict], output_dir: Path):
    """生成汇总 CSV"""
    import csv

    csv_path = output_dir / "summary.csv"

    # 按 (agent, family, instance) 聚合
    stats = defaultdict(lambda: {"makespans": [], "avg_dts": [], "total_dts": [], "dcounts": []})
    for r in records:
        if r.get("status") != "completed":
            continue
        key = (r["agent"], r["family"], r["instance"])
        stats[key]["makespans"].append(r.get("makespan", 0))
        ds = r.get("decision_stats", {})
        if isinstance(ds, dict):
            stats[key]["avg_dts"].append(ds.get("average_decision_time", 0))
            stats[key]["total_dts"].append(ds.get("total_decision_time", 0))
            stats[key]["dcounts"].append(ds.get("decision_count", 0))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "family", "instance", "n_runs", "makespan_mean", "makespan_std", "makespan_min", "makespan_max", "avg_dt_mean", "total_dt_mean", "decision_count_mean"])
        for (agent, family, instance), s in sorted(stats.items()):
            mk = s["makespans"]
            adt = s["avg_dts"]
            tdt = s["total_dts"]
            dc = s["dcounts"]
            writer.writerow([
                agent, family, instance, len(mk),
                f"{np.mean(mk):.2f}" if mk else "",
                f"{np.std(mk):.2f}" if mk else "",
                f"{min(mk):.2f}" if mk else "",
                f"{max(mk):.2f}" if mk else "",
                f"{np.mean(adt):.6f}" if adt else "",
                f"{np.mean(tdt):.4f}" if tdt else "",
                f"{np.mean(dc):.1f}" if dc else "",
            ])

    logger.info(f"保存汇总表: {csv_path}")


def plot_benchmark(args):
    experiment_id = args.experiment_id
    exp_dir = RESULTS_ROOT / experiment_id
    results_path = exp_dir / "results.jsonl"

    if not exp_dir.exists():
        logger.error(f"实验目录不存在: {exp_dir}")
        return

    # 输出目录
    output_dir = Path(args.output_dir) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    records = load_results(results_path)
    logger.info(f"加载 {len(records)} 条记录")
    completed = [r for r in records if r.get("status") == "completed"]
    logger.info(f"其中 {len(completed)} 条为完成状态")

    if not completed:
        logger.error("无已完成的记录，无法绘图")
        return

    # 发现参与实验的 agents
    available_agents = sorted(set(r["agent"] for r in completed if r["agent"] in AGENT_COLORS))

    # 根据 --agents 参数筛选
    if args.agents:
        agents = [a for a in args.agents if a in available_agents]
        unknown = set(args.agents) - set(available_agents)
        if unknown:
            logger.warning(f"以下 Agent 不在实验结果中: {unknown}")
        if not agents:
            logger.error("筛选后无可用 Agent，退出")
            return
    else:
        agents = available_agents
    logger.info(f"参与 Agent: {agents}")

    # 对每个指标绘图
    for metric in args.metrics:
        logger.info(f"\n绘制指标: {metric}")
        grouped = group_data(records, metric)
        families = sorted(grouped.keys())
        logger.info(f"数据集族: {families}")

        # 各族单独箱型图
        for family in families:
            plot_family_boxplot(family, grouped[family], metric, agents, output_dir)

        # 综合对比图
        plot_combined_boxplot(grouped, metric, agents, output_dir)

    # 汇总 CSV
    generate_summary_csv(records, output_dir)

    logger.info(f"\n绘图完成，输出目录: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark 实验结果绘图脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python scripts/plot_benchmark.py --experiment-id bench_20260519_120000
  uv run python scripts/plot_benchmark.py --experiment-id my_exp --metric makespan avg_decision_time
  uv run python scripts/plot_benchmark.py --experiment-id my_exp --agents DualDRLAgent ORToolsBatchAgent
        """,
    )

    parser.add_argument("--experiment-id", type=str, required=True, help="实验 ID（对应 benchmark_results/ 下的目录名）")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["makespan"],
        choices=["makespan", "avg_decision_time", "total_decision_time", "decision_count"],
        help="要绘制的指标 (默认: makespan)",
    )
    parser.add_argument("--agents", nargs="+", default=None, choices=list(AGENT_COLORS.keys()), help="要绘制的 Agent 集合 (默认: 全部)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="输出目录 (默认: benchmark_plots/)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    plot_benchmark(args)
