#!/usr/bin/env python3
"""Run content and evidence checks for the current thesis review manuscript."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from build_thesis_integrated import build


ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = ROOT / "docs/thesis/chapters"
FINAL = ROOT / "docs/thesis/论文终稿草案_v1.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=FINAL)
    args = parser.parse_args()
    manuscript = args.manuscript if args.manuscript.is_absolute() else ROOT / args.manuscript
    text = manuscript.read_text(encoding="utf-8")

    chapter_paths = sorted(CHAPTER_DIR.glob("论文第*章*.md"))
    assert len(chapter_paths) == 6, f"expected six chapter drafts, found {len(chapter_paths)}"
    assert all(path.stat().st_size > 0 for path in chapter_paths)

    for chapter in range(1, 7):
        assert re.search(rf"^# 第 {chapter} 章", text, re.MULTILINE), f"missing chapter {chapter}"

    forbidden = {
        "clean": "正文不得出现 clean 场景术语",
        "uncertain": "正文不得出现 uncertain 场景术语",
        "GraphGRPOAgent": "正文不得出现实现类名",
        "HGSAdaptedAgent": "正文不得出现实现类名",
        "ORToolsAgent": "正文不得出现实现类名",
        "benchmark_results": "正文不得出现实验目录",
        "results.jsonl": "正文不得出现结果文件格式",
    }
    for token, message in forbidden.items():
        assert token not in text, message

    stale_values = ("1506.20", "377.00", "1629.60", "OR-Tools（仅使用一个基准测试批次）", "共包含 84 条")
    for value in stale_values:
        assert value not in text, f"stale result remains: {value}"

    required_values = ("1497.00", "386.67", "252 条", "本文方法额外两个训练种子的稳定性审计")
    for value in required_values:
        assert value in text, f"current result missing: {value}"

    labels = re.findall(r"^\*\*表\s+([0-9]+-[0-9]+)", text, re.MULTILINE)
    assert labels == ["5-1", "5-2", "5-3"], f"table labels are {labels}"

    evidence = [
        "benchmark_plots/formal_main_frozen_balanced_comparison_v2.csv",
        "benchmark_plots/formal_main_frozen_balanced_paired_statistics_v2.csv",
        "benchmark_plots/formal_main_frozen_balanced_paired_differences_v2.csv",
        "docs/thesis/formal_artifact_manifest_v1.json",
        "docs/thesis/literature/zotero_literature.bib",
    ]
    for relative in evidence:
        assert (ROOT / relative).is_file(), f"missing evidence: {relative}"

    # Rebuild into a temporary file and require byte-identical content. This
    # prevents future source-draft edits from silently diverging from the
    # submitted review manuscript.
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = Path(directory) / "rebuilt.md"
        build(rebuilt, version="终稿草案 v1")
        assert rebuilt.read_text(encoding="utf-8") == text, "final manuscript is out of sync with chapter drafts"

    print(f"PASS: {manuscript}")
    print(f"chapters={len(chapter_paths)} tables={len(labels)} evidence={len(evidence)}")


if __name__ == "__main__":
    main()
