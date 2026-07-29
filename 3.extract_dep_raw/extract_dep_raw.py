from __future__ import annotations

import argparse
from pathlib import Path

from ltp import LTP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEGMENTATION_DICTIONARY = PROJECT_ROOT / "segmentation_words.txt"


def load_segmentation_words(path: Path) -> list[str]:
    """读取一行一词的 UTF-8 分词词典，并拒绝重复词和空词典。"""
    words: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        word = raw_line.strip()
        if not word or word.startswith("#"):
            continue
        if word in seen:
            raise ValueError(
                f"分词词典存在重复词：{path}:{line_number}: {word}"
            )
        seen.add(word)
        words.append(word)
    if not words:
        raise ValueError(f"分词词典为空：{path}")
    return words


def display_project_path(path: Path) -> str:
    """项目内路径显示为相对路径，项目外路径显示原路径。"""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 DEP 输出完整依存事实和全部原始 ATT，不做 POS 筛选。"
    )
    parser.add_argument("input", type=Path, help="每行一句的 Markdown 文本")
    parser.add_argument("output", type=Path, help="Markdown 输出文件")
    parser.add_argument("--model", default="LTP/base", help="LTP 模型名称或路径")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--segmentation-word-frequency",
        type=int,
        default=2,
        help="自定义词典分词权重（默认 2）",
    )
    parser.add_argument(
        "--segmentation-dictionary",
        type=Path,
        default=DEFAULT_SEGMENTATION_DICTIONARY,
        help=(
            "一行一词的 UTF-8 分词词典；"
            "默认使用项目根目录的 segmentation_words.txt"
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.segmentation_word_frequency < 1:
        parser.error("--segmentation-word-frequency 必须大于 0")
    return args


def clean_token(value: object) -> str:
    return str(value).strip()


def escape_table_cell(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


def analyze_sentence(
    words: list[str],
    dependency: dict[str, list[object]],
) -> dict[str, object]:
    """保存完整 DEP 图，并原样摘出所有 label=ATT 的边。"""
    words = [clean_token(word) for word in words]
    heads = [int(head) for head in dependency["head"]]
    labels = [clean_token(label) for label in dependency["label"]]

    if not (len(words) == len(heads) == len(labels)):
        raise ValueError(
            "LTP结果长度不一致："
            f"cws={len(words)}, heads={len(heads)}, labels={len(labels)}"
        )

    dependency_edges: list[dict[str, object]] = []
    raw_att: list[dict[str, object]] = []
    for dependent_index, (head, label) in enumerate(
        zip(heads, labels, strict=True)
    ):
        edge = {
            "dependent": words[dependent_index],
            "dependent_index": dependent_index,
            "head": "ROOT" if head == 0 else words[head - 1],
            "head_index": None if head == 0 else head - 1,
            "label": label,
        }
        dependency_edges.append(edge)
        if label == "ATT" and head != 0:
            raw_att.append(
                {
                    "modifier": words[dependent_index],
                    "modifier_index": dependent_index,
                    "head": words[head - 1],
                    "head_index": head - 1,
                }
            )

    return {
        "segmentation": " / ".join(words),
        "dependency_edges": dependency_edges,
        "raw_att": raw_att,
    }


def format_dependency(items: list[dict[str, object]]) -> str:
    formatted: list[str] = []
    for item in items:
        if item["head_index"] is None:
            head_text = "0:ROOT"
        else:
            head_text = f"{int(item['head_index']) + 1}:{item['head']}"
        formatted.append(
            f"{int(item['dependent_index']) + 1}:{item['dependent']} "
            f"-[{item['label']}]→ {head_text}"
        )
    return "；".join(formatted) or "无"


def format_att(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']} → {item['head']}"
        for item in items
    ) or "无"


def main() -> None:
    args = parse_args()
    segmentation_words = load_segmentation_words(
        args.segmentation_dictionary
    )

    source_lines = args.input.read_text(encoding="utf-8").splitlines()
    records = [
        (line_number, line.strip())
        for line_number, line in enumerate(source_lines, start=1)
        if line_number > 1 and line.strip()
    ]

    model = LTP(args.model)
    model.add_words(
        segmentation_words,
        freq=args.segmentation_word_frequency,
    )

    analyses: list[dict[str, object]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        result = model.pipeline(
            [sentence for _, sentence in batch],
            tasks=["cws", "dep"],
        )
        for (line_number, sentence), words, dep in zip(
            batch,
            result.cws,
            result.dep,
            strict=True,
        ):
            analysis = analyze_sentence(words, dep)
            analysis.update(
                {
                    "source_line": line_number,
                    "sentence": sentence,
                }
            )
            analyses.append(analysis)

        completed = min(start + len(batch), len(records))
        print(f"已完成 {completed}/{len(records)}")

    if len(analyses) != len(records):
        raise RuntimeError(
            f"记录数不一致：输入 {len(records)}，输出 {len(analyses)}"
        )

    raw_att_count = sum(len(item["raw_att"]) for item in analyses)
    output_lines = [
        f"# {args.input.stem} DEP 原始句法事实",
        "",
        f"- 来源：`{args.input.as_posix()}`",
        f"- 模型：`{args.model}`",
        "- LTP任务：`cws + dep`；`cws`仅作为DEP的分词前置。",
        f"- 自定义词典：`{display_project_path(args.segmentation_dictionary)}`，"
        f"共{len(segmentation_words)}词。",
        f"- 自定义词频：`{args.segmentation_word_frequency}`。",
        "- 本阶段只保存完整DEP图，并原样提取所有`label=ATT`的边。",
        "- 本阶段不运行POS，不筛选、不纠错、不补边、不重建。",
        f"- 统计：共{len(analyses)}句，DEP原始ATT {raw_att_count}条。",
        "",
        "| 原文件行号 | 原句 | 分词（cws） | 完整DEP | DEP原始ATT |",
        "|---:|---|---|---|---|",
    ]

    for analysis in analyses:
        output_lines.append(
            f"| {analysis['source_line']} | "
            f"{escape_table_cell(analysis['sentence'])} | "
            f"{escape_table_cell(analysis['segmentation'])} | "
            f"{escape_table_cell(format_dependency(analysis['dependency_edges']))} | "
            f"{escape_table_cell(format_att(analysis['raw_att']))} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        f"已写入 {args.output}：{len(analyses)}句；"
        f"DEP原始ATT{raw_att_count}条；"
        "未运行POS筛选"
    )


if __name__ == "__main__":
    main()
