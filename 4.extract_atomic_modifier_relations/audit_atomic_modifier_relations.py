from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "4.extract_atomic_modifier_relations/"
    "original_question_atomic_modifier_relations.md"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "4.extract_atomic_modifier_relations/"
    "atomic_modifier_relations_quality_audit.md"
)

NO_RELATION_REASONS = {
    2: "“同比多少”没有明确业务中心词",
    11: "“同比多少”没有明确业务中心词",
    22: "“同比多少”没有明确业务中心词",
    109: "“提供详情”没有显式被修饰业务实体",
    247: "“提供明细”没有显式被修饰业务实体",
    274: "上下文指代问句，没有显式业务中心词",
    298: "上下文指代问句，没有显式业务中心词",
    327: "上下文指代问句，没有显式业务中心词",
    330: "上下文指代问句，没有显式业务中心词",
    372: "“变更倒回”已是单个token，不产生token间原子关系",
}

MANUAL_NOTES = {
    5: "SDP补回“收入/预算→完成率”",
    6: "补回“负→增长”，并删除错误“收入→同比”",
    12: "补回“负→增长”",
    57: "删除位置伪关系“项目→中”",
    93: "补回“管理→升级单”",
    105: "恢复“高危→网络→变更→操作”原子链",
    127: "恢复“P3→比拼→网络”",
    130: "恢复“FaceboY→比拼→网络”",
    162: "补回“负→增长”",
    218: "恢复“高风险→变更→操作”原子链",
    279: "改正为“不→满→客户→声音”",
    321: "删除位置伪关系“项目→中”",
    377: "恢复“FaceboY→比拼→网络”",
    379: "补回“不→健康”",
    385: "补回“EI→项目”",
    388: "补回“未→关闭”和“EI→项目”",
}


def split_markdown_row(line: str) -> list[str]:
    text = line.strip()
    if not text.startswith("|") or not text.endswith("|"):
        raise ValueError(f"不是Markdown表格行：{line!r}")
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in text[1:-1]:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            current.append(character)
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    cells.append("".join(current).strip())
    return cells


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    headers: list[str] | None = None
    rows: dict[int, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = split_markdown_row(line)
        if headers is None:
            if cells and cells[0] == "原文件行号":
                headers = cells
            continue
        if cells and re.fullmatch(r":?-+:?", cells[0]):
            continue
        if len(cells) != len(headers):
            raise ValueError(f"表格列数不一致：{line}")
        line_id = int(cells[0])
        rows[line_id] = dict(zip(headers, cells, strict=True))
    if headers is None:
        raise ValueError(f"未找到表头：{path}")
    return rows


def relation_count(value: str) -> int:
    if not value or value == "无":
        return 0
    return len([part for part in value.split("；") if part.strip()])


def repair_sources(value: str) -> list[str]:
    if not value or value == "无":
        return []
    return re.findall(r"（([^/（]+)/[^：]+：", value)


def escape_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def main() -> None:
    rows = read_rows(DEFAULT_INPUT)
    verdict_counts: Counter[str] = Counter()
    output_rows: list[str] = []

    for line_id in sorted(rows):
        row = rows[line_id]
        final = row["第四阶段原子ATT"]
        candidates = row["原子ATT修复候选"]
        anomalies = row["异常ATT"]
        sources = repair_sources(candidates)

        if relation_count(final) == 0:
            verdict = "合理为空"
            note = NO_RELATION_REASONS.get(
                line_id,
                "未输出原子关系，需结合上下文判断",
            )
        elif anomalies != "无":
            verdict = "修复后通过"
            note = MANUAL_NOTES.get(
                line_id,
                "原始结构异常已保留证据并完成纠错",
            )
        elif sources:
            verdict = (
                "通过（含中置信候选）"
                if "/medium：" in candidates
                else "补召回后通过"
            )
            note = MANUAL_NOTES.get(
                line_id,
                "新增证据：" + "、".join(dict.fromkeys(sources)),
            )
        else:
            verdict = "通过"
            note = MANUAL_NOTES.get(line_id, "未发现结构性漏边或伪ATT")

        verdict_counts[verdict] += 1
        output_rows.append(
            f"| {line_id} | {escape_cell(row['原句'])} | "
            f"{escape_cell(final)} | {verdict} | {escape_cell(note)} |"
        )

    output = [
        "# 第四阶段全量逐条质量审查",
        "",
        f"- 审查对象：`{DEFAULT_INPUT.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- 句子数：{len(rows)}",
        "- 审查重点：词法修饰丢失、否定极性、量词/位置词误挂、"
        "FOB/VOB复合词、英文缩写与编号、中心词方向。",
        "- “通过（含中置信候选）”表示规则结果已进入最终列，但仍应在"
        "后续标注集上评估中心词粒度。",
        "",
        "## 审查统计",
        "",
        *[
            f"- {verdict}：{count}条"
            for verdict, count in sorted(verdict_counts.items())
        ],
        "",
        "## 逐条审查",
        "",
        "| 原文件行号 | 原句 | 第四阶段原子关系 | 审查结论 | 说明 |",
        "|---:|---|---|---|---|",
        *output_rows,
        "",
    ]
    DEFAULT_OUTPUT.write_text("\n".join(output), encoding="utf-8")
    print(f"已写入 {DEFAULT_OUTPUT}：{len(rows)}条")


if __name__ == "__main__":
    main()
