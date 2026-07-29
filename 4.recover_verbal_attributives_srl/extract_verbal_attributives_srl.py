from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from ltp import LTP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEGMENTATION_DICTIONARY = PROJECT_ROOT / "segmentation_words.txt"

QUESTION_OR_QUANTITY_POS = {"r", "m", "q"}
PUNCTUATION_POS = {"wp"}
NOUN_LIKE_POS = {"n", "nl", "ns", "nt", "nz", "ni", "nh", "ws"}
ATTRIBUTE_POS = {"a", "b", "j", "nd"}
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

# 时间和地点先作为独立条件，不并入动词定语，避免把范围维度带入。
MODIFIER_ARGUMENT_ROLES = {
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "ARGM-PRP",
    "ARGM-CAU",
    "ARGM-ADV",
    "ARGM-NEG",
    "ARGM-MNR",
    "ARGM-DIR",
    "ARGM-EXT",
}


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
        description=(
            "使用 POS + DEP 锚定动词ATT，使用SRL论元恢复原句中的"
            "完整动词定语。"
        )
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


def locate_token_spans(
    sentence: str,
    words: list[str],
) -> list[tuple[int, int]]:
    """将LTP token顺序对齐回原句，区间采用左闭右开。"""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        start = sentence.find(word, cursor)
        if start < 0:
            raise ValueError(
                f"token无法对齐原句：token={word!r}, cursor={cursor}, "
                f"sentence={sentence!r}"
            )
        end = start + len(word)
        spans.append((start, end))
        cursor = end
    return spans


def is_time_word(word: str, pos: str) -> bool:
    return pos == "nt" or TIME_PATTERN.fullmatch(word) is not None


def classify_atomic_att(
    modifier: str,
    modifier_pos: str,
    head: str,
    head_pos: str,
    segmentation_words: set[str],
) -> tuple[str, str, bool]:
    """与第三阶段一致的POS过滤，仅决定原子ATT是否进入完整结果。"""
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


def normalize_srl_frames(
    frames: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """按0基谓词token索引整理SRL结果，并校验论元范围。"""
    normalized: dict[int, dict[str, Any]] = {}
    for frame in frames:
        predicate_index = int(frame["index"])
        arguments: list[dict[str, object]] = []
        for argument in frame.get("arguments", []):
            if len(argument) < 4:
                continue
            role, text, start, end = argument[:4]
            arguments.append(
                {
                    "role": clean_token(role),
                    "text": clean_token(text),
                    "start": int(start),
                    "end": int(end),
                }
            )
        normalized[predicate_index] = {
            "predicate": clean_token(frame.get("predicate", "")),
            "index": predicate_index,
            "arguments": arguments,
        }
    return normalized


def punctuation_segment(
    predicate_index: int,
    words: list[str],
    pos_tags: list[str],
) -> tuple[int, int]:
    """返回谓词所在标点分句的token区间，采用左闭右开。"""
    start = 0
    for index in range(predicate_index - 1, -1, -1):
        if (
            pos_tags[index] in PUNCTUATION_POS
            or PUNCTUATION_PATTERN.fullmatch(words[index]) is not None
        ):
            start = index + 1
            break

    end = len(words)
    for index in range(predicate_index + 1, len(words)):
        if (
            pos_tags[index] in PUNCTUATION_POS
            or PUNCTUATION_PATTERN.fullmatch(words[index]) is not None
        ):
            end = index
            break
    return start, end


def find_explicit_de(
    predicate_index: int,
    head_index: int,
    words: list[str],
) -> int | None:
    """寻找谓词与中心词之间最先出现的“的”，仅作为显式边界信号。"""
    if predicate_index >= head_index:
        return None
    for index in range(predicate_index + 1, head_index):
        if words[index] == "的":
            return index
    return None


def argument_overlaps_token(
    argument: dict[str, object],
    token_index: int,
) -> bool:
    return (
        int(argument["start"])
        <= token_index
        <= int(argument["end"])
    )


def recover_verbal_attribute(
    sentence: str,
    words: list[str],
    pos_tags: list[str],
    spans: list[tuple[int, int]],
    modifier_index: int,
    head_index: int,
    srl_frame: dict[str, Any] | None,
) -> dict[str, object]:
    """用SRL论元恢复原句连续片段；不补词、不调序、不做fallback。"""
    atomic_modifier = words[modifier_index]
    head = words[head_index]
    explicit_de_index = find_explicit_de(
        modifier_index,
        head_index,
        words,
    )

    result: dict[str, object] = {
        "modifier_index": modifier_index,
        "modifier": atomic_modifier,
        "modifier_pos": pos_tags[modifier_index],
        "head_index": head_index,
        "head": head,
        "head_pos": pos_tags[head_index],
        "has_explicit_de": explicit_de_index is not None,
        "srl_frame": srl_frame,
        "target_supported": False,
        "modifier_arguments": [],
        "status": "",
        "confidence": "unresolved",
        "candidate_modifier": "",
        "recovered_modifier": "",
        "start": None,
        "end": None,
    }

    if srl_frame is None:
        result["status"] = "srl_predicate_missing"
        return result

    if clean_token(srl_frame["predicate"]) != atomic_modifier:
        result["status"] = "srl_predicate_text_mismatch"
        return result

    segment_start, segment_end = punctuation_segment(
        modifier_index,
        words,
        pos_tags,
    )
    if not (
        segment_start <= modifier_index < segment_end
        and segment_start <= head_index < segment_end
    ):
        result["status"] = "predicate_head_cross_punctuation"
        return result

    target_arguments: list[dict[str, object]] = []
    modifier_arguments: list[dict[str, object]] = []
    for argument in srl_frame["arguments"]:
        argument_start = int(argument["start"])
        argument_end = int(argument["end"])
        if argument_start < segment_start or argument_end >= segment_end:
            continue
        if argument_overlaps_token(argument, head_index):
            target_arguments.append(argument)
            continue
        if (
            str(argument["role"]) in MODIFIER_ARGUMENT_ROLES
            and argument_start < head_index
            and argument_end < head_index
        ):
            modifier_arguments.append(argument)

    result["target_supported"] = bool(target_arguments)
    result["modifier_arguments"] = modifier_arguments

    if not modifier_arguments:
        result["status"] = "srl_no_modifier_arguments"
        return result

    selected_indices = {modifier_index}
    for argument in modifier_arguments:
        selected_indices.update(
            range(
                int(argument["start"]),
                int(argument["end"]) + 1,
            )
        )

    start_token = min(selected_indices)
    content_end_token = max(selected_indices)
    if start_token >= head_index:
        result["status"] = "modifier_not_before_head"
        return result

    end_token = (
        explicit_de_index
        if explicit_de_index is not None
        else content_end_token
    )
    if end_token >= head_index:
        end_token = head_index - 1
    if end_token < modifier_index:
        result["status"] = "invalid_modifier_boundary"
        return result

    # 连续片段不能跨越标点；论元虽然可离散，但最终文本不拼接。
    for index in range(start_token, end_token + 1):
        if (
            pos_tags[index] in PUNCTUATION_POS
            or PUNCTUATION_PATTERN.fullmatch(words[index]) is not None
        ):
            result["status"] = "recovered_span_cross_punctuation"
            return result

    start_char = spans[start_token][0]
    end_char = spans[end_token][1]
    recovered_modifier = sentence[start_char:end_char].strip()
    if not recovered_modifier:
        result["status"] = "empty_recovered_span"
        return result
    if recovered_modifier not in sentence:
        raise AssertionError(
            f"恢复结果不是原句片段：{recovered_modifier!r}, "
            f"sentence={sentence!r}"
        )

    result["start"] = start_char
    result["end"] = end_char
    result["candidate_modifier"] = recovered_modifier
    target_supported = bool(target_arguments)
    has_explicit_de = explicit_de_index is not None

    if target_supported and has_explicit_de:
        result["confidence"] = "high"
        result["status"] = "recovered_explicit"
        result["recovered_modifier"] = recovered_modifier
    elif target_supported:
        result["confidence"] = "medium"
        result["status"] = "recovered_implicit"
        result["recovered_modifier"] = recovered_modifier
    elif has_explicit_de:
        result["confidence"] = "medium"
        result["status"] = "recovered_without_target_support"
        result["recovered_modifier"] = recovered_modifier
    else:
        result["confidence"] = "low"
        result["status"] = "candidate_without_target_support"
    return result


def analyze_sentence(
    sentence: str,
    words: list[str],
    pos_tags: list[str],
    dependency: dict[str, list[object]],
    srl_frames: list[dict[str, Any]],
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

    spans = locate_token_spans(sentence, words)
    srl_by_predicate = normalize_srl_frames(srl_frames)
    candidates: list[dict[str, object]] = []
    final_atomic_att: list[dict[str, object]] = []

    for modifier_index, (head, label) in enumerate(
        zip(heads, labels, strict=True)
    ):
        if label != "ATT" or head == 0:
            continue

        head_index = head - 1
        item = {
            "modifier_index": modifier_index,
            "modifier": words[modifier_index],
            "modifier_pos": pos_tags[modifier_index],
            "head_index": head_index,
            "head": words[head_index],
            "head_pos": pos_tags[head_index],
        }
        _, _, keep = classify_atomic_att(
            item["modifier"],
            item["modifier_pos"],
            item["head"],
            item["head_pos"],
            segmentation_words,
        )
        if not keep:
            continue
        final_atomic_att.append(item)

        if pos_tags[modifier_index].startswith("v"):
            candidates.append(
                recover_verbal_attribute(
                    sentence,
                    words,
                    pos_tags,
                    spans,
                    modifier_index,
                    head_index,
                    srl_by_predicate.get(modifier_index),
                )
            )

    accepted_by_relation = {
        (int(candidate["modifier_index"]), int(candidate["head_index"])): str(
            candidate["recovered_modifier"]
        )
        for candidate in candidates
        if candidate["recovered_modifier"]
    }
    complete_att: list[dict[str, object]] = []
    for item in final_atomic_att:
        relation_key = (
            int(item["modifier_index"]),
            int(item["head_index"]),
        )
        complete_item = dict(item)
        recovered_modifier = accepted_by_relation.get(relation_key)
        if recovered_modifier:
            complete_item["modifier"] = recovered_modifier
            complete_item["source"] = "srl_recovered"
        else:
            complete_item["source"] = "atomic_att"
        complete_att.append(complete_item)

    return {
        "segmentation": " / ".join(words),
        "word_pos": "；".join(
            f"{word}/{pos}"
            for word, pos in zip(words, pos_tags, strict=True)
        ),
        "final_atomic_att": final_atomic_att,
        "candidates": candidates,
        "complete_att": complete_att,
    }


def format_atomic(candidates: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']}/{item['modifier_pos']} → "
        f"{item['head']}/{item['head_pos']}"
        for item in candidates
    ) or "无"


def format_plain_att(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']} → {item['head']}"
        for item in items
    ) or "无"


def format_srl_evidence(candidates: list[dict[str, object]]) -> str:
    formatted: list[str] = []
    for item in candidates:
        frame = item["srl_frame"]
        if frame is None:
            formatted.append(
                f"{item['modifier']}@{int(item['modifier_index']) + 1}：未匹配"
            )
            continue
        arguments = "，".join(
            f"{argument['role']}={argument['text']}"
            for argument in frame["arguments"]
        ) or "无论元"
        formatted.append(
            f"{frame['predicate']}@{int(frame['index']) + 1}"
            f"（{arguments}）"
        )
    return "；".join(formatted) or "无"


def format_decisions(candidates: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']} → {item['head']}："
        f"{item['status']}，confidence={item['confidence']}，"
        f"explicit_de={'yes' if item['has_explicit_de'] else 'no'}，"
        f"target_supported={'yes' if item['target_supported'] else 'no'}"
        for item in candidates
    ) or "无"


def format_recovered(candidates: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['recovered_modifier']} → {item['head']}"
        for item in candidates
        if item["recovered_modifier"]
    ) or "无"


def format_candidate_spans(candidates: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['candidate_modifier']} → {item['head']}"
        for item in candidates
        if item["candidate_modifier"]
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
            tasks=["cws", "pos", "dep", "srl"],
        )
        for (
            (line_number, sentence),
            words,
            pos_tags,
            dep,
            srl,
        ) in zip(
            batch,
            result.cws,
            result.pos,
            result.dep,
            result.srl,
            strict=True,
        ):
            analysis = analyze_sentence(
                sentence,
                words,
                pos_tags,
                dep,
                srl,
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

    all_candidates = [
        candidate
        for analysis in analyses
        for candidate in analysis["candidates"]
    ]
    candidate_span_count = sum(
        bool(candidate["candidate_modifier"])
        for candidate in all_candidates
    )
    accepted_recovery_count = sum(
        bool(candidate["recovered_modifier"])
        for candidate in all_candidates
    )
    complete_relation_count = sum(
        len(analysis["complete_att"])
        for analysis in analyses
    )

    output_lines = [
        f"# {args.input.stem} SRL辅助动词定语恢复",
        "",
        f"- 来源：`{args.input.as_posix()}`",
        f"- 模型：`{args.model}`",
        "- LTP任务：`cws + pos + dep + srl`。",
        f"- 自定义词典：`{display_project_path(args.segmentation_dictionary)}`，"
        f"共{len(segmentation_words)}词。",
        f"- 自定义词频：`{args.segmentation_word_frequency}`。",
        "- DEP作用：只以动词性`ATT`作为待恢复关系锚点。",
        "- SRL作用：匹配同token谓词，使用论元确定连续原句范围。",
        "- “的”的作用：仅作为显式边界和置信度信号，不是必要条件。",
        "- 恢复约束：只截取原句连续字符，不补词、不调序、不做语义改写。",
        "- SRL缺失：保留原子ATT并标记，不启用DEP子树fallback。",
        "- 接受规则：仅high/medium进入恢复结果；low只保留为审计候选。",
        "- 完整结果：继承第三阶段全部POS过滤后原子ATT；"
        "SRL接受恢复时替换对应短动词ATT。",
        "- 当前阶段：不做维度过滤、关系合并、SDP辅助或Schema映射。",
        f"- 统计：动词ATT候选{len(all_candidates)}条，"
        f"形成连续候选{candidate_span_count}条，"
        f"接受恢复{accepted_recovery_count}条，"
        f"完整结果共{complete_relation_count}条关系。",
        "",
        "| 原文件行号 | 原句 | 分词（cws） | 词性（pos） | "
        "第三阶段原子ATT | DEP动词ATT | SRL证据 | 恢复判定 | "
        "SRL连续候选 | 接受的动词定语 | 第四阶段完整结果 |",
        "|---:|---|---|---|---|---|---|---|---|---|---|",
    ]

    for analysis in analyses:
        candidates = analysis["candidates"]
        output_lines.append(
            f"| {analysis['source_line']} | "
            f"{escape_table_cell(analysis['sentence'])} | "
            f"{escape_table_cell(analysis['segmentation'])} | "
            f"{escape_table_cell(analysis['word_pos'])} | "
            f"{escape_table_cell(format_plain_att(analysis['final_atomic_att']))} | "
            f"{escape_table_cell(format_atomic(candidates))} | "
            f"{escape_table_cell(format_srl_evidence(candidates))} | "
            f"{escape_table_cell(format_decisions(candidates))} | "
            f"{escape_table_cell(format_candidate_spans(candidates))} | "
            f"{escape_table_cell(format_recovered(candidates))} | "
            f"{escape_table_cell(format_plain_att(analysis['complete_att']))} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        f"已写入 {args.output}：{len(analyses)}句；"
        f"动词ATT候选{len(all_candidates)}条，"
        f"形成连续候选{candidate_span_count}条，"
        f"接受恢复{accepted_recovery_count}条，"
        f"完整结果{complete_relation_count}条关系"
    )


if __name__ == "__main__":
    main()
