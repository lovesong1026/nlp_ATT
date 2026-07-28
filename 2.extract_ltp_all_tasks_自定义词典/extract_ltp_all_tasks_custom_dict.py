from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ltp import LTP


ALL_TASKS = ["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"]

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
        description="使用自定义分词词典运行 LTP 全部七项任务。"
    )
    parser.add_argument("input", type=Path, help="每行一句的 Markdown 文本")
    parser.add_argument("output", type=Path, help="Markdown 输出文件")
    parser.add_argument("--model", default="LTP/base", help="LTP 模型名称或路径")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--segmentation-word-frequency",
        type=int,
        default=2,
        help="注册自定义词时使用的词频权重（默认 2）",
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


def escape_table_cell(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


def clean_word(word: object) -> str:
    return str(word).strip()


def format_word_pos(words: list[str], pos_tags: list[str]) -> str:
    return "；".join(
        f"{clean_word(word)}/{tag}"
        for word, tag in zip(words, pos_tags, strict=True)
    )


def format_ner(entities: list[Any]) -> str:
    formatted: list[str] = []
    for entity in entities:
        if isinstance(entity, dict):
            tag = entity.get("tag", entity.get("label", "?"))
            text = entity.get("text", entity.get("entity", "?"))
            start = entity.get("start", "?")
            end = entity.get("end", "?")
        elif isinstance(entity, (tuple, list)) and len(entity) >= 4:
            tag, text, start, end = entity[:4]
        else:
            formatted.append(str(entity))
            continue
        formatted.append(
            f"{text}/{tag}[token {int(start) + 1}-{int(end) + 1}]"
        )
    return "；".join(formatted) or "无"


def format_srl(items: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for item in items:
        predicate = item.get("predicate", "?")
        predicate_index = int(item.get("index", -1)) + 1
        arguments: list[str] = []
        for argument in item.get("arguments", []):
            if len(argument) < 4:
                arguments.append(str(argument))
                continue
            tag, text, start, end = argument[:4]
            arguments.append(
                f"{tag}={text}[token {int(start) + 1}-{int(end) + 1}]"
            )
        argument_text = "，".join(arguments) or "无论元"
        formatted.append(
            f"{predicate}@token {predicate_index}（{argument_text}）"
        )
    return "；".join(formatted) or "无"


def format_dependency(
    words: list[str],
    dependency: dict[str, list[Any]],
) -> str:
    heads = dependency["head"]
    labels = dependency["label"]
    if not (len(words) == len(heads) == len(labels)):
        raise ValueError(
            "依存结果长度不一致："
            f"words={len(words)}, heads={len(heads)}, labels={len(labels)}"
        )

    formatted: list[str] = []
    for index, (word, head, label) in enumerate(
        zip(words, heads, labels, strict=True),
        start=1,
    ):
        head_index = int(head)
        head_text = "ROOT" if head_index == 0 else clean_word(words[head_index - 1])
        formatted.append(
            f"{index}:{clean_word(word)} -[{label}]→ "
            f"{head_index}:{head_text}"
        )
    return "；".join(formatted) or "无"


def format_sdpg(words: list[str], edges: list[Any]) -> str:
    formatted: list[str] = []
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) < 3:
            formatted.append(str(edge))
            continue
        source, target, label = edge[:3]
        source_index = int(source)
        target_index = int(target)
        source_text = (
            "ROOT"
            if source_index == 0
            else clean_word(words[source_index - 1])
        )
        target_text = (
            "ROOT"
            if target_index == 0
            else clean_word(words[target_index - 1])
        )
        formatted.append(
            f"{source_index}:{source_text} -[{label}]→ "
            f"{target_index}:{target_text}"
        )
    return "；".join(formatted) or "无"


def main() -> None:
    args = parse_args()
    segmentation_words = load_segmentation_words(
        args.segmentation_dictionary
    )
    source_lines = args.input.read_text(encoding="utf-8").splitlines()
    records = [
        (line_number, text.strip())
        for line_number, text in enumerate(source_lines, start=1)
        if line_number > 1 and text.strip()
    ]

    model = LTP(args.model)
    model.add_words(
        segmentation_words,
        freq=args.segmentation_word_frequency,
    )
    analyses: list[dict[str, str | int]] = []

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        result = model.pipeline(
            [sentence for _, sentence in batch],
            tasks=ALL_TASKS,
        )

        for (
            (line_number, sentence),
            words,
            pos_tags,
            ner,
            srl,
            dep,
            sdp,
            sdpg,
        ) in zip(
            batch,
            result.cws,
            result.pos,
            result.ner,
            result.srl,
            result.dep,
            result.sdp,
            result.sdpg,
            strict=True,
        ):
            clean_words = [clean_word(word) for word in words]
            analyses.append(
                {
                    "source_line": line_number,
                    "sentence": sentence,
                    "cws": " / ".join(clean_words),
                    "pos": format_word_pos(clean_words, pos_tags),
                    "ner": format_ner(ner),
                    "srl": format_srl(srl),
                    "dep": format_dependency(clean_words, dep),
                    "sdp": format_dependency(clean_words, sdp),
                    "sdpg": format_sdpg(clean_words, sdpg),
                }
            )

        completed = min(start + len(batch), len(records))
        print(f"已完成 {completed}/{len(records)}")

    if len(analyses) != len(records):
        raise RuntimeError(
            f"记录数不一致：输入 {len(records)}，输出 {len(analyses)}"
        )

    output_lines = [
        f"# {args.input.stem} LTP 全任务输出（自定义词典）",
        "",
        f"- 来源：`{args.input.as_posix()}`",
        f"- 模型：`{args.model}`",
        "- LTP任务：`cws + pos + ner + srl + dep + sdp + sdpg`。",
        f"- 自定义词典：`{display_project_path(args.segmentation_dictionary)}`，"
        f"共{len(segmentation_words)}词。",
        f"- 自定义词频：`{args.segmentation_word_frequency}`。",
        "- 词典作用：只保护分词边界，不进行语义改写。",
        "- token序号：从1开始；`0:ROOT`表示根节点。",
        "",
        "| 原文件行号 | 原句 | cws | pos | ner | srl | dep | sdp | sdpg |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]

    for analysis in analyses:
        output_lines.append(
            "| "
            + " | ".join(
                [
                    str(analysis["source_line"]),
                    escape_table_cell(analysis["sentence"]),
                    escape_table_cell(analysis["cws"]),
                    escape_table_cell(analysis["pos"]),
                    escape_table_cell(analysis["ner"]),
                    escape_table_cell(analysis["srl"]),
                    escape_table_cell(analysis["dep"]),
                    escape_table_cell(analysis["sdp"]),
                    escape_table_cell(analysis["sdpg"]),
                ]
            )
            + " |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"已写入 {args.output}：{len(analyses)} 句")


if __name__ == "__main__":
    main()
