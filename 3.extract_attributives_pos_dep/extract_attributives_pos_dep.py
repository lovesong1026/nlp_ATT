from __future__ import annotations

import argparse
import re
from pathlib import Path

from ltp import LTP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEGMENTATION_DICTIONARY = PROJECT_ROOT / "segmentation_words.txt"

NOUN_LIKE_POS = {"n", "nl", "ns", "nt", "nz", "ni", "nh", "ws"}
ATTRIBUTE_POS = {"a", "b", "j", "nd"}
QUESTION_OR_QUANTITY_POS = {"r", "m", "q"}

PUNCTUATION_PATTERN = re.compile(r"^[，,。！？!?；;：:、]$")
YEAR_PATTERN = re.compile(r"^(?:\d{4}|本|当|去|今|明)年$")
MONTH_PATTERN = re.compile(r"^(?:\d{1,2}|[一二三四五六七八九十]+)月$")
TIME_PATTERN = re.compile(
    r"^(?:"
    r"\d{4}年|"
    r"\d{1,2}月|"
    r"\d{4}年\d{1,2}月|"
    r"当前|目前|本年|当年|今年|去年|明年|"
    r"H[12]"
    r")$",
    re.IGNORECASE,
)
NOISE_WORD_PATTERN = re.compile(
    r"^(?:"
    r"这|这些|这个|该|此|上述|前述|"
    r"多少|哪个|哪些|哪|几|什么|"
    r"\d+|[一二两三四五六七八九十百]+|"
    r"个|起|次|张|项|条|份"
    r")$"
)


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
        description="使用 POS + DEP 提取原子 ATT 定中关系，不做重建。"
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


def is_time_word(word: str, pos: str) -> bool:
    return pos == "nt" or TIME_PATTERN.fullmatch(word) is not None


def classify_att(
    modifier: str,
    modifier_pos: str,
    head: str,
    head_pos: str,
    segmentation_words: set[str],
) -> tuple[str, str, bool]:
    """仅使用词性和词面过滤明显噪声，不修正依存关系。"""
    if (
        YEAR_PATTERN.fullmatch(modifier) is not None
        and MONTH_PATTERN.fullmatch(head) is not None
    ):
        return "过滤", "日历内部修饰", False

    if (
        head not in segmentation_words
        and head_pos in QUESTION_OR_QUANTITY_POS
    ):
        return "过滤", f"中心词为疑问或数量词/{head_pos}", False

    if (
        PUNCTUATION_PATTERN.fullmatch(modifier) is not None
        or modifier_pos in {"wp", "u"}
    ):
        return "过滤", "标点或结构助词", False

    if NOISE_WORD_PATTERN.fullmatch(modifier) is not None:
        return "过滤", "疑问、指示或数量噪声", False

    if modifier in segmentation_words:
        return "保留", "自定义词典业务词", True

    if modifier_pos in QUESTION_OR_QUANTITY_POS:
        return "过滤", f"疑问或数量词性/{modifier_pos}", False

    if is_time_word(modifier, modifier_pos):
        return "过滤", "时间修饰", False

    if modifier_pos in NOUN_LIKE_POS:
        return "保留", f"名词性/{modifier_pos}", True

    if modifier_pos in ATTRIBUTE_POS:
        return "保留", f"属性性/{modifier_pos}", True

    if modifier_pos.startswith("v"):
        return "保留", f"动词性修饰/{modifier_pos}", True

    return "保留", f"其他内容词/{modifier_pos}", True


def analyze_sentence(
    words: list[str],
    pos_tags: list[str],
    dependency: dict[str, list[object]],
    segmentation_words: set[str],
) -> dict[str, object]:
    words = [clean_token(word) for word in words]
    pos_tags = [clean_token(pos) for pos in pos_tags]
    heads = [int(head) for head in dependency["head"]]
    labels = [clean_token(label) for label in dependency["label"]]

    if not (len(words) == len(pos_tags) == len(heads) == len(labels)):
        raise ValueError(
            "LTP结果长度不一致："
            f"cws={len(words)}, pos={len(pos_tags)}, "
            f"heads={len(heads)}, labels={len(labels)}"
        )

    raw_att: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    final_att: list[dict[str, object]] = []

    for modifier_index, (head, label) in enumerate(
        zip(heads, labels, strict=True)
    ):
        if label != "ATT" or head == 0:
            continue

        head_index = head - 1
        item = {
            "modifier": words[modifier_index],
            "modifier_pos": pos_tags[modifier_index],
            "head": words[head_index],
            "head_pos": pos_tags[head_index],
            "modifier_index": modifier_index,
            "head_index": head_index,
        }
        raw_att.append(item)

        action, reason, keep = classify_att(
            item["modifier"],
            item["modifier_pos"],
            item["head"],
            item["head_pos"],
            segmentation_words,
        )
        decisions.append(
            {
                **item,
                "action": action,
                "reason": reason,
            }
        )
        if keep:
            final_att.append(item)

    return {
        "segmentation": " / ".join(words),
        "word_pos": "；".join(
            f"{word}/{pos}"
            for word, pos in zip(words, pos_tags, strict=True)
        ),
        "raw_att": raw_att,
        "decisions": decisions,
        "final_att": final_att,
    }


def format_att(items: list[dict[str, object]], *, with_pos: bool) -> str:
    if with_pos:
        return "；".join(
            f"{item['modifier']}/{item['modifier_pos']} → "
            f"{item['head']}/{item['head_pos']}"
            for item in items
        ) or "无"
    return "；".join(
        f"{item['modifier']} → {item['head']}"
        for item in items
    ) or "无"


def format_decisions(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['action']}[{item['reason']}]："
        f"{item['modifier']}/{item['modifier_pos']} → "
        f"{item['head']}/{item['head_pos']}"
        for item in items
    ) or "无"


def main() -> None:
    args = parse_args()
    segmentation_words = load_segmentation_words(
        args.segmentation_dictionary
    )
    segmentation_word_set = set(segmentation_words)

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
            tasks=["cws", "pos", "dep"],
        )
        for (line_number, sentence), words, pos_tags, dep in zip(
            batch,
            result.cws,
            result.pos,
            result.dep,
            strict=True,
        ):
            analysis = analyze_sentence(
                words,
                pos_tags,
                dep,
                segmentation_word_set,
            )
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

    output_lines = [
        f"# {args.input.stem} POS + DEP 原子定中关系",
        "",
        f"- 来源：`{args.input.as_posix()}`",
        f"- 模型：`{args.model}`",
        "- LTP任务：`cws + pos + dep`；`cws`仅作为POS和DEP的分词前置。",
        f"- 自定义词典：`{display_project_path(args.segmentation_dictionary)}`，"
        f"共{len(segmentation_words)}词。",
        f"- 自定义词频：`{args.segmentation_word_frequency}`。",
        "- 提取范围：只提取DEP标签为`ATT`的原子定中关系。",
        "- POS作用：只过滤时间、数量、疑问、指示和结构词等明显噪声。",
        "- 不执行：完整定语重建、中心词提升、局部关系补充、"
        "维度过滤、关系合并、语义改写。",
        "",
        "| 原文件行号 | 原句 | 分词（cws） | 词性（pos） | "
        "DEP原始ATT | POS判定 | 最终原子ATT |",
        "|---:|---|---|---|---|---|---|",
    ]

    for analysis in analyses:
        output_lines.append(
            f"| {analysis['source_line']} | "
            f"{escape_table_cell(analysis['sentence'])} | "
            f"{escape_table_cell(analysis['segmentation'])} | "
            f"{escape_table_cell(analysis['word_pos'])} | "
            f"{escape_table_cell(format_att(analysis['raw_att'], with_pos=True))} | "
            f"{escape_table_cell(format_decisions(analysis['decisions']))} | "
            f"{escape_table_cell(format_att(analysis['final_att'], with_pos=False))} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        f"已写入 {args.output}：{len(analyses)} 句；"
        f"词典{len(segmentation_words)}词，"
        f"freq={args.segmentation_word_frequency}"
    )


if __name__ == "__main__":
    main()
