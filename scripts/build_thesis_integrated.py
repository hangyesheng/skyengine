#!/usr/bin/env python3
"""Build a single review manuscript from the six current chapter drafts."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = [
    ROOT / "docs/thesis/chapters/论文第1章_绪论草稿.md",
    ROOT / "docs/thesis/chapters/论文第2章_相关工作草稿.md",
    ROOT / "docs/thesis/chapters/论文第3章_平台设计与实现草稿.md",
    ROOT / "docs/thesis/chapters/论文第4章_GraphGRPO协同调度方法草稿.md",
    ROOT / "docs/thesis/chapters/论文第5章_实验设计与结果分析草稿.md",
    ROOT / "docs/thesis/chapters/论文第6章_结论与展望草稿.md",
]


def build(output: Path, version: str = "v2") -> None:
    missing = [path for path in CHAPTERS if not path.exists()]
    if missing:
        raise FileNotFoundError("missing chapter draft: " + ", ".join(str(path) for path in missing))

    sections = [path.read_text(encoding="utf-8").strip() for path in CHAPTERS]
    header = f"""# 柔性制造系统中加工—AGV物流协同调度与动态事件适应研究

> 整合稿版本：{version}（{date.today().isoformat()}）
>
> 本文件由当前第 1—6 章草稿生成，用于连续阅读和论文内容审阅。正式提交前仍需按照学校模板统一处理参考文献、图表、公式、页眉页脚和格式编号。

## 章节结构

1. 绪论
2. 相关工作
3. 平台设计与实现
4. 基于关系图与组相对优势的加工—物流协同调度方法
5. 实验设计与结果分析
6. 结论与展望

---

"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + "\n\n---\n\n".join(sections) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/thesis/论文整合稿_v2.md")
    parser.add_argument("--version", default="v2", help="写入整合稿页眉的版本标识")
    args = parser.parse_args()
    build(args.output, args.version)


if __name__ == "__main__":
    main()
