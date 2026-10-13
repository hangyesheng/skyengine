#!/usr/bin/env python3
"""Generate publication-style figures from frozen thesis summaries.

The figures use only the frozen main-method and generalization summaries:
本文方法, HGS 适配实现, and time-limited OR-Tools.  Error bars in the
generalization figures are bootstrap 95% intervals; the formal-test figure
uses one standard deviation to show instance-level variation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


METHODS = ["本文方法", "HGS适配实现", "OR-Tools"]
METHOD_LABELS = {"本文方法": "本文方法", "HGS适配实现": "HGS适配实现", "OR-Tools": "OR-Tools"}
COLORS = {"本文方法": "#2F5597", "HGS适配实现": "#70AD47", "OR-Tools": "#ED7D31"}
FAMILIES = ["barnes", "behnke", "dauzere", "fattahi"]
FAMILY_LABELS = {"barnes": "Barnes", "behnke": "Behnke", "dauzere": "Dauzere", "fattahi": "Fattahi"}


def configure_font() -> None:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_makespan(summary: list[dict], scenario: str, output: Path) -> None:
    rows = [r for r in summary if r["scenario"] == scenario]
    lookup = {(r["method"], r["family"]): r for r in rows}
    x = np.arange(len(FAMILIES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for index, method in enumerate(METHODS):
        values = [float(lookup[(method, family)]["makespan_mean"]) for family in FAMILIES]
        lows = [float(lookup[(method, family)]["makespan_ci95_low"]) for family in FAMILIES]
        highs = [float(lookup[(method, family)]["makespan_ci95_high"]) for family in FAMILIES]
        errors = [np.array(values) - np.array(lows), np.array(highs) - np.array(values)]
        bars = ax.bar(x + (index - 1) * width, values, width, label=METHOD_LABELS[method], color=COLORS[method], yerr=errors, capsize=3, edgecolor="white", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [FAMILY_LABELS[f] for f in FAMILIES])
    ax.set_ylabel(r"总完工时间 $C_{\max}$")
    ax.set_title("跨数据集族泛化：" + ("基础场景" if scenario == "base" else "动态事件场景"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, output)


def plot_generalization_degradation(rows: list[dict], output: Path) -> None:
    lookup = {(r["method"], r["family"]): r for r in rows}
    x = np.arange(len(FAMILIES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for index, method in enumerate(METHODS):
        values = [100 * float(lookup[(method, family)]["relative_mean"]) for family in FAMILIES]
        lows = [100 * float(lookup[(method, family)]["relative_ci95_low"]) for family in FAMILIES]
        highs = [100 * float(lookup[(method, family)]["relative_ci95_high"]) for family in FAMILIES]
        errors = [np.array(values) - np.array(lows), np.array(highs) - np.array(values)]
        bars = ax.bar(x + (index - 1) * width, values, width, label=METHOD_LABELS[method], color=COLORS[method], yerr=errors, capsize=3, edgecolor="white", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [FAMILY_LABELS[f] for f in FAMILIES])
    ax.set_ylabel(r"相对退化率 $D_{C_{\max}}$ (%)")
    ax.set_title("跨数据集族动态性能退化")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, output)


def plot_formal_main(comparison: list[dict], output: Path) -> None:
    families = ["brandimarte", "kacem"]
    labels = {"brandimarte": "Brandimarte", "kacem": "Kacem"}
    def normalize_method(value: str) -> str:
        if value.startswith("本文方法"):
            return "本文方法"
        if value.startswith("HGS"):
            return "HGS适配实现"
        return "OR-Tools"

    lookup = {(normalize_method(r["method"]), r["family"]): r for r in comparison}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=False)
    x = np.arange(len(METHODS))
    for ax, family in zip(axes, families):
        base = [float(lookup[(method, family)]["base_makespan_mean"]) for method in METHODS]
        event = [float(lookup[(method, family)]["event_makespan_mean"]) for method in METHODS]
        base_std = [float(lookup[(method, family)]["base_makespan_std"]) for method in METHODS]
        event_std = [float(lookup[(method, family)]["event_makespan_std"]) for method in METHODS]
        width = 0.34
        ax.bar(x - width / 2, base, width, yerr=base_std, capsize=3, label="基础场景", color="#A9C4E4", edgecolor="white")
        ax.bar(x + width / 2, event, width, yerr=event_std, capsize=3, label="动态事件场景", color="#F4B183", edgecolor="white")
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_title(labels[family])
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=18)
    axes[0].set_ylabel(r"总完工时间 $C_{\max}$")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("正式测试集的基础场景与动态事件场景对比", y=1.02)
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("benchmark_plots"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/thesis/figures"))
    args = parser.parse_args()
    configure_font()
    generalization = read_csv(args.root / "generalization_frozen_lb_v1_summary.csv")
    degradation = read_csv(args.root / "generalization_frozen_lb_v1_degradation_summary.csv")
    formal = read_csv(args.root / "formal_main_frozen_balanced_comparison_v2.csv")
    plot_generalization_makespan(generalization, "base", args.output_dir / "fig_generalization_base_makespan.png")
    plot_generalization_makespan(generalization, "dynamic", args.output_dir / "fig_generalization_dynamic_makespan.png")
    plot_generalization_degradation(degradation, args.output_dir / "fig_generalization_relative_degradation.png")
    plot_formal_main(formal, args.output_dir / "fig_formal_main_scenario_comparison.png")
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
