from __future__ import annotations

import argparse
import importlib.util
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ltp import LTP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEGMENTATION_DICTIONARY = PROJECT_ROOT / "segmentation_words.txt"
DEFAULT_DIMENSION_INPUT = (
    PROJECT_ROOT / "data/dimension_extracted_question.md"
)
MERGE_RULES_SCRIPT = (
    PROJECT_ROOT
    / "4.1.merge_atomic_modifier_relations"
    / "merge_atomic_modifier_relations.py"
)

QUESTION_OR_QUANTITY_POS = {"r", "m", "q"}
PUNCTUATION_POS = {"wp"}
NOUN_LIKE_POS = {"n", "nl", "ns", "nt", "nz", "ni", "nh", "ws"}
ATTRIBUTE_POS = {"a", "b", "j", "nd"}
FUNCTION_POS = {"wp", "u", "p", "c", "d"}
NEGATION_WORDS = {"不", "未", "无", "非", "没"}
COORDINATION_WORDS = {"和", "与", "及", "或", "以及", "或者", "、"}
POSITIONAL_FUNCTION_WORDS = {"中", "内"}
NON_ENTITY_HEAD_WORDS = {"同比"}
GENERIC_PREDICATE_WORDS = {
    "有",
    "还有",
    "共有",
    "是",
    "为",
    "存在",
    "发生",
    "完成",
    "提供",
    "执行",
    "涉及",
    "导致",
    "进行",
    "占",
    "属于",
    "等于",
    "大于",
    "小于",
    "高于",
    "低于",
    "超过",
    "少于",
    "多于",
    "超期",
}
ALPHANUMERIC_TOKEN_PATTERN = re.compile(r"(?=.*[A-Za-z0-9])[A-Za-z0-9_-]+$")
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
METRIC_HEAD_PATTERN = re.compile(
    r"^(?:数|数量|次数|占比|比例|比率|完成率|成功率|及时率|"
    r"成本率|利润率|销毛率|成熟度|.*率)$"
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


def load_merge_rules_module() -> Any:
    """加载第4.1阶段的确定性合并规则，避免复制两套实现。"""
    spec = importlib.util.spec_from_file_location(
        "stage4_1_merge_atomic_modifier_relations",
        MERGE_RULES_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载规则合并代码：{MERGE_RULES_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用 POS、DEP局部结构和SRL证据筛选、纠错并补召回"
            "原子ATT；不重建完整定语。"
        )
    )
    parser.add_argument("input", type=Path, help="每行一句的 Markdown 文本")
    parser.add_argument("output", type=Path, help="Markdown 输出文件")
    parser.add_argument(
        "--model",
        default="LTP/base2",
        help="LTP 模型名称或路径（默认 LTP/base2）",
    )
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
    parser.add_argument(
        "--dimension-input",
        type=Path,
        default=DEFAULT_DIMENSION_INPUT,
        help=(
            "与原文件按物理行号对齐的维度提取后问题；"
            "默认使用 data/dimension_extracted_question.md，"
            "空行或缺失行按空结果处理"
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


def normalize_with_positions(text: str) -> tuple[str, list[int]]:
    """移除空白，并保留规范化字符到原句字符位置的映射。"""
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char)
        positions.append(index)
    return "".join(chars), positions


def align_excluded_dimension_positions(
    original: str,
    dimension_sentence: str,
    *,
    missing_as_empty: bool,
) -> dict[str, object]:
    """通过纯删除对齐定位原句中已被抽取的维度字符。"""
    normalized_original, original_positions = normalize_with_positions(
        original
    )
    normalized_dimension, _ = normalize_with_positions(dimension_sentence)

    if not normalized_dimension:
        excluded_positions = set(original_positions)
        status = "缺失按空处理" if missing_as_empty else "空结果"
    else:
        matcher = SequenceMatcher(
            None,
            normalized_original,
            normalized_dimension,
            autojunk=False,
        )
        retained_normalized: set[int] = set()
        invalid_operations: list[str] = []
        for (
            tag,
            original_start,
            original_end,
            dimension_start,
            dimension_end,
        ) in matcher.get_opcodes():
            if tag == "equal":
                retained_normalized.update(
                    range(original_start, original_end)
                )
            elif tag != "delete":
                invalid_operations.append(
                    f"{tag}:{original_start}-{original_end}/"
                    f"{dimension_start}-{dimension_end}"
                )
        if invalid_operations:
            raise ValueError(
                "维度后问题不是原句的纯删除结果："
                f"original={original!r}, "
                f"dimension={dimension_sentence!r}, "
                f"operations={invalid_operations}"
            )
        excluded_positions = {
            original_positions[index]
            for index in range(len(original_positions))
            if index not in retained_normalized
        }
        status = (
            "未删除维度"
            if not excluded_positions
            else "已对齐"
        )

    return {
        "dimension_sentence": dimension_sentence,
        "dimension_status": status,
        "excluded_positions": excluded_positions,
    }


def filter_atomic_relations_by_dimensions(
    relations: list[dict[str, object]],
    token_spans: list[tuple[int, int]],
    excluded_positions: set[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """修饰词或中心词与已删除维度重叠时，过滤整条原子关系。"""
    kept: list[dict[str, object]] = []
    removed: list[dict[str, object]] = []
    for relation in relations:
        modifier_index = int(relation["modifier_index"])
        head_index = int(relation["head_index"])
        modifier_start, modifier_end = token_spans[modifier_index]
        head_start, head_end = token_spans[head_index]
        modifier_overlaps = any(
            modifier_start <= position < modifier_end
            for position in excluded_positions
        )
        head_overlaps = any(
            head_start <= position < head_end
            for position in excluded_positions
        )
        if modifier_overlaps or head_overlaps:
            removed.append(
                {
                    **relation,
                    "dimension_filter_reason": (
                        "修饰词和中心词均为已抽取维度"
                        if modifier_overlaps and head_overlaps
                        else (
                            "修饰词为已抽取维度"
                            if modifier_overlaps
                            else "中心词为已抽取维度"
                        )
                    ),
                }
            )
        else:
            kept.append(relation)
    return kept, removed


def is_time_word(word: str, pos: str) -> bool:
    return pos == "nt" or TIME_PATTERN.fullmatch(word) is not None


def classify_atomic_att(
    modifier: str,
    modifier_pos: str,
    head: str,
    head_pos: str,
    segmentation_words: set[str],
) -> tuple[str, str, bool]:
    """第四阶段的POS/词面筛选，只决定直接DEP-ATT是否进入候选结果。"""
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

    if modifier in POSITIONAL_FUNCTION_WORDS:
        return "过滤", "位置结构词", False

    if head in POSITIONAL_FUNCTION_WORDS:
        return "过滤", "中心词为位置结构词", False

    if head in NON_ENTITY_HEAD_WORDS:
        return "过滤", "中心词为比较或统计运算词", False

    if (
        head in GENERIC_PREDICATE_WORDS
        and head not in segmentation_words
    ):
        return "过滤", "中心词为通用谓词，不是业务实体", False

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


def is_content_modifier(
    word: str,
    pos: str,
    segmentation_words: set[str],
) -> bool:
    """判断token能否成为原子ATT修饰语，不依赖业务词表。"""
    if word in segmentation_words:
        return True
    if (
        word in POSITIONAL_FUNCTION_WORDS
        or
        pos in FUNCTION_POS
        or pos in QUESTION_OR_QUANTITY_POS
        or is_time_word(word, pos)
        or PUNCTUATION_PATTERN.fullmatch(word) is not None
        or NOISE_WORD_PATTERN.fullmatch(word) is not None
        or word in NEGATION_WORDS
    ):
        return False
    return True


def is_content_head(
    word: str,
    pos: str,
    segmentation_words: set[str],
) -> bool:
    """判断token能否承接原子ATT。"""
    if word in segmentation_words:
        return True
    return not (
        word in POSITIONAL_FUNCTION_WORDS
        or
        word in GENERIC_PREDICATE_WORDS
        or
        pos in FUNCTION_POS
        or pos in QUESTION_OR_QUANTITY_POS
        or PUNCTUATION_PATTERN.fullmatch(word) is not None
        or NOISE_WORD_PATTERN.fullmatch(word) is not None
    )


def contains_boundary(
    start_index: int,
    end_index: int,
    words: list[str],
    pos_tags: list[str],
    *,
    include_de: bool,
) -> bool:
    """判断两个token之间是否跨越标点或可选的“的”边界。"""
    start, end = sorted((start_index, end_index))
    for index in range(start + 1, end):
        if (
            pos_tags[index] in PUNCTUATION_POS
            or PUNCTUATION_PATTERN.fullmatch(words[index]) is not None
            or (include_de and words[index] == "的")
        ):
            return True
    return False


def make_atomic_relation(
    modifier_index: int,
    head_index: int,
    words: list[str],
    pos_tags: list[str],
    source: str,
    confidence: str,
    reason: str,
    original_label: str | None = None,
) -> dict[str, object]:
    return {
        "modifier_index": modifier_index,
        "modifier": words[modifier_index],
        "modifier_pos": pos_tags[modifier_index],
        "head_index": head_index,
        "head": words[head_index],
        "head_pos": pos_tags[head_index],
        "source": source,
        "confidence": confidence,
        "reason": reason,
        "original_label": original_label,
    }


def add_relation(
    relations: dict[tuple[int, int], dict[str, object]],
    relation: dict[str, object],
) -> None:
    """按token位置去重；直接DEP关系优先于同位置修复关系。"""
    key = (
        int(relation["modifier_index"]),
        int(relation["head_index"]),
    )
    existing = relations.get(key)
    if existing is None or (
        existing["source"] != "dep_att"
        and relation["source"] == "dep_att"
    ):
        relations[key] = relation


def find_right_content_child(
    parent_index: int,
    modifier_index: int,
    words: list[str],
    pos_tags: list[str],
    heads: list[int],
    labels: list[str],
    segmentation_words: set[str],
) -> int | None:
    """寻找动词右侧由其支配的名词性业务中心词。"""
    candidates: list[int] = []
    for index, (head, label) in enumerate(zip(heads, labels, strict=True)):
        if (
            head - 1 == parent_index
            and index > parent_index
            and label in {"ATT", "VOB", "FOB", "DBL"}
            and is_content_head(
                words[index],
                pos_tags[index],
                segmentation_words,
            )
            and not contains_boundary(
                modifier_index,
                index,
                words,
                pos_tags,
                include_de=False,
            )
        ):
            candidates.append(index)
    return max(candidates) if candidates else None


def resolve_modifier_target(
    modifier_index: int,
    words: list[str],
    pos_tags: list[str],
    heads: list[int],
    labels: list[str],
    segmentation_words: set[str],
    *,
    allow_verb_child: bool = False,
) -> int | None:
    """沿局部依存结构寻找属性词真正修饰的名词中心词。"""
    head_index = heads[modifier_index] - 1
    if head_index < 0:
        return None

    # 属性词被误分析成动词的ADV/VOB/FOB等，而该动词本身ATT到名词。
    if pos_tags[head_index].startswith("v"):
        upper_head = heads[head_index] - 1
        if (
            upper_head >= 0
            and labels[head_index] == "ATT"
            and is_content_head(
                words[upper_head],
                pos_tags[upper_head],
                segmentation_words,
            )
        ):
            if METRIC_HEAD_PATTERN.fullmatch(words[upper_head]):
                return head_index
            return upper_head

        # 量词可能被错挂到动词，而真正名词是动词右侧宾语。
        if allow_verb_child:
            child = find_right_content_child(
                head_index,
                modifier_index,
                words,
                pos_tags,
                heads,
                labels,
                segmentation_words,
            )
            if child is not None:
                return child

    # 内容词ATT到量词，量词再连接真正业务中心词。
    if (
        pos_tags[head_index] in QUESTION_OR_QUANTITY_POS
        and words[head_index] not in segmentation_words
    ):
        upper_head = heads[head_index] - 1
        if upper_head >= 0:
            if (
                is_content_head(
                    words[upper_head],
                    pos_tags[upper_head],
                    segmentation_words,
                )
                and not pos_tags[upper_head].startswith("v")
            ):
                return upper_head
            if pos_tags[upper_head].startswith("v"):
                child = find_right_content_child(
                    upper_head,
                    modifier_index,
                    words,
                    pos_tags,
                    heads,
                    labels,
                    segmentation_words,
                )
                if child is not None:
                    return child
    return None


def right_nominal_head_before_boundary(
    modifier_index: int,
    words: list[str],
    pos_tags: list[str],
    segmentation_words: set[str],
    *,
    max_distance: int = 4,
) -> int | None:
    """为反向ATT寻找右侧、同一短语内的中心词。"""
    candidates: list[int] = []
    end = min(len(words), modifier_index + max_distance + 1)
    for index in range(modifier_index + 1, end):
        if (
            pos_tags[index] in PUNCTUATION_POS
            or PUNCTUATION_PATTERN.fullmatch(words[index]) is not None
        ):
            break
        if words[index] == "的":
            break
        if (
            is_content_head(
                words[index],
                pos_tags[index],
                segmentation_words,
            )
            and (
                pos_tags[index].startswith("n")
                or pos_tags[index] in {"k", "v"}
            )
        ):
            candidates.append(index)
    return max(candidates) if candidates else None


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


def srl_corrected_head(
    modifier_index: int,
    head_index: int,
    words: list[str],
    pos_tags: list[str],
    heads: list[int],
    labels: list[str],
    srl_frame: dict[str, Any] | None,
    segmentation_words: set[str],
) -> int | None:
    """利用显式“的”和SRL目标论元纠正过短或指标化中心词。"""
    de_index = find_explicit_de(modifier_index, head_index, words)
    if srl_frame is None or de_index is None:
        return None

    target_arguments = [
        argument
        for argument in srl_frame["arguments"]
        if argument_overlaps_token(argument, head_index)
    ]
    if not target_arguments:
        return None

    candidate_indices: list[int] = []
    for argument in target_arguments:
        argument_start = int(argument["start"])
        argument_end = int(argument["end"])
        start = max(de_index + 1, argument_start)
        end = argument_end
        if any(
            (
                words[index] in COORDINATION_WORDS
                or pos_tags[index] == "c"
                or pos_tags[index] in PUNCTUATION_POS
                or PUNCTUATION_PATTERN.fullmatch(words[index]) is not None
            )
            for index in range(start, end + 1)
        ):
            continue

        # 指标词经常吸收前面的事件中心，例如“未关闭的管理升级数量”。
        # 此时谓词真正作用于指标前最后一个内容中心，而不是“数量/率”。
        if METRIC_HEAD_PATTERN.fullmatch(words[head_index]):
            search_range = range(start, head_index)
        else:
            # 非指标短中心则向SRL目标论元右端提升，例如
            # “关闭→管理”提升为“关闭→升级单”。
            search_range = range(head_index + 1, end + 1)

        local_candidates: list[int] = []
        for index in search_range:
            if (
                is_content_head(
                    words[index],
                    pos_tags[index],
                    segmentation_words,
                )
                and not METRIC_HEAD_PATTERN.fullmatch(words[index])
                and not contains_boundary(
                    head_index,
                    index,
                    words,
                    pos_tags,
                    include_de=True,
                )
            ):
                local_candidates.append(index)
        if local_candidates:
            candidate_indices.append(max(local_candidates))

    return max(candidate_indices) if candidate_indices else None


def build_atomic_att_repairs(
    words: list[str],
    pos_tags: list[str],
    heads: list[int],
    labels: list[str],
    direct_att: list[dict[str, object]],
    srl_by_predicate: dict[int, dict[str, Any]],
    segmentation_words: set[str],
    sdp_heads: list[int] | None = None,
    sdp_labels: list[str] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """补充和纠正DEP漏掉或挂错的原子ATT，不重建完整定语。"""
    relations: dict[tuple[int, int], dict[str, object]] = {}
    anomalies: list[dict[str, object]] = []
    repairs: list[dict[str, object]] = []
    replaced_direct_keys: set[tuple[int, int]] = set()
    if sdp_heads is None:
        sdp_heads = [0] * len(words)
    if sdp_labels is None:
        sdp_labels = [""] * len(words)
    if not (len(sdp_heads) == len(sdp_labels) == len(words)):
        raise ValueError(
            "SDP结果长度不一致："
            f"cws={len(words)}, heads={len(sdp_heads)}, "
            f"labels={len(sdp_labels)}"
        )

    def record_repair(
        modifier_index: int,
        head_index: int,
        source: str,
        confidence: str,
        reason: str,
        original_label: str | None = None,
        allow_semantic_operator: bool = False,
    ) -> None:
        modifier_is_valid = is_content_modifier(
            words[modifier_index],
            pos_tags[modifier_index],
            segmentation_words,
        ) or (
            allow_semantic_operator
            and words[modifier_index] in NEGATION_WORDS
        )
        if (
            modifier_index == head_index
            or not modifier_is_valid
            or not is_content_head(
                words[head_index],
                pos_tags[head_index],
                segmentation_words,
            )
            or contains_boundary(
                modifier_index,
                head_index,
                words,
                pos_tags,
                include_de=False,
            )
        ):
            return
        key = (modifier_index, head_index)
        if key in relations:
            return
        relation = make_atomic_relation(
            modifier_index,
            head_index,
            words,
            pos_tags,
            source,
            confidence,
            reason,
            original_label,
        )
        if key not in {
            (
                int(item["modifier_index"]),
                int(item["head_index"]),
            )
            for item in repairs
        }:
            repairs.append(relation)
        add_relation(relations, relation)

    def modifier_has_relation(modifier_index: int) -> bool:
        return any(key[0] == modifier_index for key in relations)

    # 先接收方向正常的直接DEP-ATT；反向关系只作为异常证据。
    for item in direct_att:
        modifier_index = int(item["modifier_index"])
        head_index = int(item["head_index"])
        if modifier_index > head_index:
            anomalies.append(
                {
                    **item,
                    "anomaly": "backward_att",
                    "reason": "修饰语位于中心词之后，疑似挂错中心词",
                }
            )
            continue

        corrected_head = None
        # “场景的风险项目数”一类结构中，显式“的”左侧修饰语可能
        # 直接越过业务实体挂到指标词。若“的”后的非指标词本身直接
        # ATT到该指标，选择最靠左的这一局部中心，避免仅凭词面猜测。
        de_index = find_explicit_de(modifier_index, head_index, words)
        if (
            de_index is not None
            and METRIC_HEAD_PATTERN.fullmatch(words[head_index])
            and not pos_tags[modifier_index].startswith("v")
        ):
            metric_children = [
                index
                for index in range(de_index + 1, head_index)
                if (
                    heads[index] - 1 == head_index
                    and labels[index] == "ATT"
                    and is_content_head(
                        words[index],
                        pos_tags[index],
                        segmentation_words,
                    )
                    and not METRIC_HEAD_PATTERN.fullmatch(words[index])
                )
            ]
            if metric_children:
                corrected_metric_head = min(metric_children)
                replaced_direct_keys.add((modifier_index, head_index))
                anomalies.append(
                    {
                        **item,
                        "anomaly": "explicit_de_metric_head",
                        "reason": (
                            "显式“的”左侧修饰语越过局部实体挂到指标词"
                        ),
                    }
                )
                record_repair(
                    modifier_index,
                    corrected_metric_head,
                    "explicit_de_metric_head",
                    "high",
                    "沿指标词的直接ATT子节点恢复显式“的”后的局部中心",
                    "ATT",
                )
                continue

        # “不满客户声音”一类结构中，LTP可能输出“满→声音”，同时把
        # “客户”作为“满”的VOB。否定词+右侧宾语共同表明局部中心词
        # 应先落在宾语，再由宾语连接外层名词。
        object_children = [
            index
            for index, (child_head, child_label) in enumerate(
                zip(heads, labels, strict=True)
            )
            if (
                child_head - 1 == modifier_index
                and child_label in {"VOB", "FOB"}
                and modifier_index < index < head_index
            )
        ]
        if (
            modifier_index > 0
            and words[modifier_index - 1] in NEGATION_WORDS
            and object_children
        ):
            object_index = min(object_children)
            replaced_direct_keys.add((modifier_index, head_index))
            anomalies.append(
                {
                    **item,
                    "anomaly": "negated_object_head",
                    "reason": (
                        "否定谓词带右侧宾语，原ATT跨过了更近的名词中心"
                    ),
                }
            )
            record_repair(
                modifier_index,
                object_index,
                "negated_object_repair",
                "high",
                "否定谓词先修饰其右侧宾语",
                labels[object_index],
            )
            record_repair(
                object_index,
                head_index,
                "negated_object_repair",
                "high",
                "局部宾语继续修饰外层名词中心",
                labels[object_index],
            )
            continue

        if str(item["modifier_pos"]).startswith("v"):
            corrected_head = srl_corrected_head(
                modifier_index,
                head_index,
                words,
                pos_tags,
                heads,
                labels,
                srl_by_predicate.get(modifier_index),
                segmentation_words,
            )
        if corrected_head is not None:
            replaced_direct_keys.add((modifier_index, head_index))
            anomalies.append(
                {
                    **item,
                    "anomaly": "short_head",
                    "reason": (
                        "SRL目标论元表明中心词过短或被指标词吸收，"
                        f"建议提升到{words[corrected_head]}"
                    ),
                }
            )
            record_repair(
                modifier_index,
                corrected_head,
                "srl_head_lift",
                "high",
                "显式“的”+SRL目标论元共同支持局部中心词提升",
                "ATT",
            )
            continue

        relation = dict(item)
        relation.update(
            {
                "source": "dep_att",
                "confidence": "high",
                "reason": "LTP直接输出的正向ATT",
                "original_label": "ATT",
            }
        )
        add_relation(relations, relation)

    # 内容词被挂到量词，或属性词被误标为动词内部关系。
    for modifier_index, label in enumerate(labels):
        if not is_content_modifier(
            words[modifier_index],
            pos_tags[modifier_index],
            segmentation_words,
        ):
            continue

        head_index = heads[modifier_index] - 1
        if head_index < 0:
            continue

        if (
            pos_tags[head_index] in QUESTION_OR_QUANTITY_POS
            and words[head_index] not in segmentation_words
        ):
            target = resolve_modifier_target(
                modifier_index,
                words,
                pos_tags,
                heads,
                labels,
                segmentation_words,
                allow_verb_child=True,
            )
            if target is not None:
                anomalies.append(
                    {
                        "modifier_index": modifier_index,
                        "modifier": words[modifier_index],
                        "modifier_pos": pos_tags[modifier_index],
                        "head_index": head_index,
                        "head": words[head_index],
                        "head_pos": pos_tags[head_index],
                        "anomaly": "quantity_head",
                        "reason": "内容修饰语错误挂到量词或疑问词",
                    }
                )
                record_repair(
                    modifier_index,
                    target,
                    "quantity_head_lift",
                    "high",
                    "跳过错误量词中心词，提升到同一短语的内容中心词",
                    label,
                )

        if (
            pos_tags[modifier_index] in ATTRIBUTE_POS
            and label != "ATT"
            and not contains_boundary(
                modifier_index,
                head_index,
                words,
                pos_tags,
                include_de=True,
            )
        ):
            target = resolve_modifier_target(
                modifier_index,
                words,
                pos_tags,
                heads,
                labels,
                segmentation_words,
            )
            # “高风险变更操作”中上层“操作”也常被标成动词；此时保留
            # 更局部的“高风险→变更”，再由“变更→操作”组成原子链。
            if (
                target is not None
                and target != head_index
                and pos_tags[target].startswith("v")
                and sdp_heads[modifier_index] - 1 == head_index
                and sdp_labels[modifier_index] in {"FEAT", "dFEAT"}
            ):
                target = head_index
            if (
                target is None
                and pos_tags[head_index].startswith("v")
                and labels[head_index] in {"SBV", "VOB", "FOB", "DBL"}
                and modifier_index < head_index
                and any(
                    modifier_index < index < head_index
                    and heads[index] - 1 == head_index
                    and labels[index] == "ATT"
                    for index in range(len(words))
                )
            ):
                target = head_index
            if target is not None and not contains_boundary(
                modifier_index,
                target,
                words,
                pos_tags,
                include_de=True,
            ):
                record_repair(
                    modifier_index,
                    target,
                    "compact_np",
                    "medium",
                    (
                        f"属性词被标为{label}，但其依存路径位于"
                        "无“的”紧凑名词短语内"
                    ),
                    label,
                )

    # SDP 的 FEAT/dFEAT 是稳定的语义特征边，可补 DEP 将定中关系误标为
    # FOB/VOB 等情况。这里只接收左修饰、右中心且不跨边界的内容词。
    for modifier_index, (sdp_head, sdp_label) in enumerate(
        zip(sdp_heads, sdp_labels, strict=True)
    ):
        head_index = sdp_head - 1
        if (
            sdp_label not in {"FEAT", "dFEAT"}
            or head_index < 0
            or modifier_index >= head_index
            or modifier_has_relation(modifier_index)
            or not is_content_modifier(
                words[modifier_index],
                pos_tags[modifier_index],
                segmentation_words,
            )
            or not is_content_head(
                words[head_index],
                pos_tags[head_index],
                segmentation_words,
            )
            or contains_boundary(
                modifier_index,
                head_index,
                words,
                pos_tags,
                include_de=True,
            )
        ):
            continue
        record_repair(
            modifier_index,
            head_index,
            "sdp_feature",
            "high",
            f"SDP-{sdp_label}将左侧内容词识别为右侧中心词的语义特征",
            sdp_label,
        )

    # LTP会把没有显式连词的紧凑复合短语误标成COO，例如把
    # “服务订货”“管理升级”分析为并列。真正并列通常有“和/与/或/、”
    # 等边界；无标记、相邻且至少一项为动词性名词时，按词序恢复局部
    # 原子链。若两项同时挂到同一个更远中心，只保留“前项→后项→中心”
    # 的局部结构，避免平行长边掩盖复合词内部关系。
    for modifier_index, label in enumerate(labels):
        if label != "COO":
            continue
        sibling_index = heads[modifier_index] - 1
        if (
            sibling_index < 0
            or abs(modifier_index - sibling_index) != 1
            or not (
                pos_tags[modifier_index].startswith("v")
                or pos_tags[sibling_index].startswith("v")
            )
            or words[modifier_index] in GENERIC_PREDICATE_WORDS
            or words[sibling_index] in GENERIC_PREDICATE_WORDS
        ):
            continue
        left_index = min(modifier_index, sibling_index)
        right_index = max(modifier_index, sibling_index)
        if (
            not is_content_modifier(
                words[left_index],
                pos_tags[left_index],
                segmentation_words,
            )
            or not is_content_head(
                words[right_index],
                pos_tags[right_index],
                segmentation_words,
            )
            or any(
                words[index] in COORDINATION_WORDS
                or pos_tags[index] == "c"
                for index in range(left_index + 1, right_index)
            )
        ):
            continue

        record_repair(
            left_index,
            right_index,
            "unmarked_coordination_compound",
            "high",
            "相邻动词性内容词被误标为无连词并列，按词序恢复紧凑复合关系",
            label,
        )

        for (existing_modifier, existing_head) in list(relations):
            if (
                existing_modifier == left_index
                and existing_head > right_index
                and (right_index, existing_head) in relations
            ):
                replaced_direct_keys.add(
                    (existing_modifier, existing_head)
                )
                anomalies.append(
                    {
                        **relations[(existing_modifier, existing_head)],
                        "anomaly": "unmarked_coordination_long_edge",
                        "reason": (
                            "无连词紧凑复合短语已有更局部的逐层中心关系"
                        ),
                    }
                )

    # SRL把连续片段整体识别为同一个核心论元时，如果相邻内容词被DEP
    # 平行挂到论元外谓词，或平行挂到论元内的名词化谓词，说明局部定中
    # 结构被外层谓词吸收。只在同一连续核心论元、相邻且共享依存中心时
    # 恢复前项到后项，避免对普通句子相邻词无条件造边。
    for frame in srl_by_predicate.values():
        predicate_index = int(frame["index"])
        for argument in frame["arguments"]:
            if str(argument["role"]) not in {
                "A0",
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
            }:
                continue
            argument_start = int(argument["start"])
            argument_end = int(argument["end"])
            if argument_start < 0 or argument_end >= len(words):
                continue
            for left_index in range(argument_start, argument_end):
                right_index = left_index + 1
                if (
                    modifier_has_relation(left_index)
                    or words[left_index] in COORDINATION_WORDS
                    or words[right_index] in COORDINATION_WORDS
                    or pos_tags[left_index] == "c"
                    or pos_tags[right_index] == "c"
                    or words[left_index] in NEGATION_WORDS
                    or words[right_index] in NEGATION_WORDS
                    or words[right_index] in GENERIC_PREDICATE_WORDS
                    or right_index == predicate_index
                    or not is_content_modifier(
                        words[left_index],
                        pos_tags[left_index],
                        segmentation_words,
                    )
                    or not is_content_head(
                        words[right_index],
                        pos_tags[right_index],
                        segmentation_words,
                    )
                ):
                    continue

                shared_head = heads[left_index] - 1
                if shared_head < 0 or heads[right_index] - 1 != shared_head:
                    continue
                shared_external_predicate = (
                    shared_head == predicate_index
                    and not (
                        argument_start
                        <= predicate_index
                        <= argument_end
                    )
                    and pos_tags[right_index].startswith("v")
                )
                shared_internal_nominal_predicate = (
                    argument_start
                    <= shared_head
                    <= argument_end
                    and shared_head > right_index
                    and pos_tags[shared_head].startswith("v")
                    and labels[left_index] in {"SBV", "VOB", "FOB", "DBL"}
                    and labels[right_index] in {"SBV", "VOB", "FOB", "DBL"}
                    and (
                        shared_head + 1 >= len(words)
                        or words[shared_head + 1] != "的"
                    )
                )
                if not (
                    shared_external_predicate
                    or shared_internal_nominal_predicate
                ):
                    continue
                record_repair(
                    left_index,
                    right_index,
                    "srl_argument_compound",
                    "medium",
                    (
                        "相邻内容词属于同一SRL核心论元且被DEP平行挂到"
                        "同一谓词，恢复论元内部局部修饰"
                    ),
                    labels[left_index],
                )

    # 在SRL核心论元内部，“名词 + 名词化动词 + 名词中心”常被分析为
    # 主谓结构。若左词直接依存到紧邻右侧谓词，且右侧没有“的”引出
    # 关系从句，则保留局部名词化修饰；显式关系从句仍交给SRL恢复。
    for frame in srl_by_predicate.values():
        for argument in frame["arguments"]:
            if str(argument["role"]) not in {
                "A0",
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
            }:
                continue
            argument_start = int(argument["start"])
            argument_end = int(argument["end"])
            for left_index in range(argument_start, argument_end):
                right_index = left_index + 1
                if (
                    modifier_has_relation(left_index)
                    or heads[left_index] - 1 != right_index
                    or labels[left_index] not in {"SBV", "VOB", "FOB", "DBL"}
                    or not pos_tags[right_index].startswith("v")
                    or words[right_index] in GENERIC_PREDICATE_WORDS
                    or any(
                        words[index] == "的"
                        for index in range(
                            right_index + 1,
                            argument_end + 1,
                        )
                    )
                    or not is_content_modifier(
                        words[left_index],
                        pos_tags[left_index],
                        segmentation_words,
                    )
                ):
                    continue
                record_repair(
                    left_index,
                    right_index,
                    "srl_nominalized_predicate",
                    "medium",
                    "SRL核心论元内名词紧邻名词化谓词，恢复局部修饰关系",
                    labels[left_index],
                )

    # 动词性名词与右侧名词宾语同时位于同一个SRL核心论元时，DEP常把
    # “交付 EI 项目”分析成普通“交付→项目”谓宾。若二者之间只有直接
    # ATT到该名词的紧凑修饰语，则将其保留为名词化复合关系；跨“的”、
    # 连词、标点或核心论元边界时不触发。
    for frame in srl_by_predicate.values():
        for argument in frame["arguments"]:
            if str(argument["role"]) not in {
                "A0",
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
            }:
                continue
            argument_start = int(argument["start"])
            argument_end = int(argument["end"])
            for verb_index in range(argument_start, argument_end):
                if (
                    not pos_tags[verb_index].startswith("v")
                    or words[verb_index] in GENERIC_PREDICATE_WORDS
                    or labels[verb_index] == "HED"
                ):
                    continue
                noun_children = [
                    index
                    for index in range(verb_index + 1, argument_end + 1)
                    if (
                        heads[index] - 1 == verb_index
                        and labels[index] in {"VOB", "FOB", "DBL"}
                        and (
                            pos_tags[index].startswith("n")
                            or pos_tags[index] == "ws"
                        )
                        and all(
                            (
                                heads[between] - 1 == index
                                and labels[between] == "ATT"
                            )
                            for between in range(verb_index + 1, index)
                        )
                        and not contains_boundary(
                            verb_index,
                            index,
                            words,
                            pos_tags,
                            include_de=True,
                        )
                        and not any(
                            words[after] == "的"
                            for after in range(index + 1, argument_end + 1)
                        )
                    )
                ]
                if not noun_children:
                    continue
                noun_index = min(noun_children)
                record_repair(
                    verb_index,
                    noun_index,
                    "srl_verbal_nominal_object",
                    "medium",
                    (
                        "动词性名词及其紧凑名词宾语位于同一SRL核心论元，"
                        "保留名词化复合关系"
                    ),
                    labels[noun_index],
                )

                # 若前置属性词只是因为同挂外层谓词而临时连接到该
                # 动词性名词，最终中心不是指标词时，将属性投射到实体
                # 中心，避免把局部事件链当成最终实体修饰。
                for (attribute_index, event_index), relation in list(
                    relations.items()
                ):
                    if (
                        event_index != verb_index
                        or relation["source"] != "srl_argument_compound"
                        or pos_tags[attribute_index] not in ATTRIBUTE_POS
                        or METRIC_HEAD_PATTERN.fullmatch(words[noun_index])
                    ):
                        continue
                    replaced_direct_keys.add(
                        (attribute_index, event_index)
                    )
                    record_repair(
                        attribute_index,
                        noun_index,
                        "srl_attribute_entity_projection",
                        "medium",
                        "SRL紧凑论元中的属性词投射到名词化事件的实体中心",
                        str(relation.get("original_label") or ""),
                    )

    # 保留否定极性。它不是传统 DEP-ATT，但对“未关闭、不健康”等业务
    # 状态不可丢失，因此作为语义原子关系输出。
    for modifier_index, word in enumerate(words):
        if word not in NEGATION_WORDS:
            continue
        head_index = heads[modifier_index] - 1
        sdp_head_index = sdp_heads[modifier_index] - 1
        if (
            sdp_labels[modifier_index] == "mNEG"
            and sdp_head_index >= 0
        ):
            head_index = sdp_head_index
        if (
            head_index >= 0
            and abs(head_index - modifier_index) <= 2
            and not contains_boundary(
                modifier_index,
                head_index,
                words,
                pos_tags,
                include_de=False,
            )
        ):
            record_repair(
                modifier_index,
                head_index,
                "semantic_polarity",
                "high",
                "否定词与局部谓词形成不可丢失的业务状态原子",
                sdp_labels[modifier_index] or labels[modifier_index],
                allow_semantic_operator=True,
            )

    # LTP常把“负增长、及时恢复、疲劳驾驶、同比增加”分析成 ADV。
    # 当属性词紧邻非通用谓词时，恢复词法修饰关系；已有更具体中心词关系
    # 时不重复添加另一中心词。
    for modifier_index, label in enumerate(labels):
        head_index = heads[modifier_index] - 1
        if (
            label != "ADV"
            or head_index != modifier_index + 1
            or pos_tags[modifier_index] not in ATTRIBUTE_POS
            or not pos_tags[head_index].startswith("v")
            or words[head_index] in GENERIC_PREDICATE_WORDS
            or words[modifier_index] in NEGATION_WORDS
            or modifier_has_relation(modifier_index)
        ):
            continue
        record_repair(
            modifier_index,
            head_index,
            "adjacent_lexical_modifier",
            "high",
            "属性词紧邻非通用谓词，DEP-ADV表示词法化业务修饰",
            label,
        )

    # 名词等内容词被标成 FOB，但其中心动词继续 ATT 到名词时，将修饰语
    # 提升到名词中心；中心本身是名词时直接补边。
    for modifier_index, label in enumerate(labels):
        if label != "FOB" or not is_content_modifier(
            words[modifier_index],
            pos_tags[modifier_index],
            segmentation_words,
        ):
            continue
        head_index = heads[modifier_index] - 1
        if head_index < 0 or words[head_index] in GENERIC_PREDICATE_WORDS:
            continue
        target: int | None = None
        if pos_tags[head_index].startswith("v"):
            upper_head = heads[head_index] - 1
            if (
                upper_head >= 0
                and labels[head_index] == "ATT"
                and is_content_head(
                    words[upper_head],
                    pos_tags[upper_head],
                    segmentation_words,
                )
            ):
                target = (
                    head_index
                    if pos_tags[upper_head].startswith("v")
                    else upper_head
                )
        elif is_content_head(
            words[head_index],
            pos_tags[head_index],
            segmentation_words,
        ):
            target = head_index
        if target is not None:
            record_repair(
                modifier_index,
                target,
                "fob_nominal_compound",
                "high",
                "DEP-FOB位于名词化业务短语内，恢复到名词中心",
                label,
            )

    # 英文缩写/编号常不在词典中，并被误判成谓词论元。若出现
    # “代码 + 动词性名词 + 名词”紧凑结构，恢复代码到动词性名词、
    # 动词性名词到最终名词两条原子关系。
    for verb_index in range(1, len(words) - 1):
        code_index = verb_index - 1
        noun_index = verb_index + 1
        if (
            ALPHANUMERIC_TOKEN_PATTERN.fullmatch(words[code_index]) is None
            or not pos_tags[verb_index].startswith("v")
            or not (
                pos_tags[noun_index].startswith("n")
                or pos_tags[noun_index] == "ws"
            )
            or heads[noun_index] - 1 != verb_index
            or labels[noun_index] not in {"VOB", "FOB", "DBL"}
            or words[verb_index] in GENERIC_PREDICATE_WORDS
        ):
            continue
        record_repair(
            code_index,
            verb_index,
            "alphanumeric_compound",
            "high",
            "英文缩写或编号位于紧凑业务复合词首部",
            labels[code_index],
        )
        record_repair(
            verb_index,
            noun_index,
            "verbal_nominal_compound",
            "high",
            "动词性名词与右侧名词构成紧凑业务复合词",
            labels[noun_index],
        )

    # “交付 EI 项目”中，缩写可能被误挂成动词宾语。若该动词继续 ATT
    # 到右侧名词，则将缩写提升到同一个名词中心。
    for code_index, word in enumerate(words):
        if (
            ALPHANUMERIC_TOKEN_PATTERN.fullmatch(word) is None
            or modifier_has_relation(code_index)
            or labels[code_index] not in {"VOB", "FOB"}
        ):
            continue
        verb_index = heads[code_index] - 1
        if (
            verb_index < 0
            or not pos_tags[verb_index].startswith("v")
            or labels[verb_index] != "ATT"
        ):
            continue
        noun_index = heads[verb_index] - 1
        if (
            noun_index >= 0
            and code_index < noun_index
            and is_content_head(
                words[noun_index],
                pos_tags[noun_index],
                segmentation_words,
            )
        ):
            record_repair(
                code_index,
                noun_index,
                "alphanumeric_head_lift",
                "high",
                "英文缩写被误挂为动词宾语，提升到同一复合短语的名词中心",
                labels[code_index],
            )

    # 显式“的”前的属性组若未形成右向ATT，连接到“的”后的名词短语根。
    for de_index, word in enumerate(words):
        if word != "的" or de_index == 0:
            continue
        left_index = de_index - 1
        if not is_content_modifier(
            words[left_index],
            pos_tags[left_index],
            segmentation_words,
        ):
            continue

        group = {left_index}
        for index, label in enumerate(labels):
            head_index = heads[index] - 1
            if label == "COO" and (
                index in group or head_index in group
            ):
                group.add(index)
                if head_index >= 0:
                    group.add(head_index)

        # 已经沿局部依存路径ATT到“的”右侧时，不重复造边。
        already_supported = False
        for member in group:
            current = member
            visited: set[int] = set()
            for _ in range(4):
                if current in visited:
                    break
                visited.add(current)
                head_index = heads[current] - 1
                if head_index < 0:
                    break
                if labels[current] == "ATT" and head_index > de_index:
                    already_supported = True
                    break
                current = head_index
            if already_supported:
                break
        if already_supported:
            continue

        right_candidates: list[int] = []
        for index in range(
            de_index + 1,
            min(len(words), de_index + 9),
        ):
            if (
                pos_tags[index] in PUNCTUATION_POS
                or PUNCTUATION_PATTERN.fullmatch(words[index]) is not None
            ):
                break
            if (
                pos_tags[index].startswith("n")
                or pos_tags[index] in {"k", "v"}
            ) and is_content_head(
                words[index],
                pos_tags[index],
                segmentation_words,
            ):
                right_candidates.append(index)
        if not right_candidates:
            continue
        non_metric_candidates = [
            index
            for index in right_candidates
            if not METRIC_HEAD_PATTERN.fullmatch(words[index])
        ]
        target = (
            max(non_metric_candidates)
            if non_metric_candidates
            else max(right_candidates)
        )
        for member in sorted(group):
            if (
                member >= de_index
                or not is_content_modifier(
                    words[member],
                    pos_tags[member],
                    segmentation_words,
                )
            ):
                continue
            record_repair(
                member,
                target,
                "explicit_de_head",
                "medium",
                "显式“的”前属性组未形成右向ATT，连接到右侧名词短语根",
                labels[member],
            )

    # 并列修饰语共享同一中心词。
    for modifier_index, label in enumerate(labels):
        if label != "COO" or not is_content_modifier(
            words[modifier_index],
            pos_tags[modifier_index],
            segmentation_words,
        ):
            continue
        sibling_index = heads[modifier_index] - 1
        if sibling_index < 0:
            continue
        sibling_head = heads[sibling_index] - 1
        target: int | None = None
        if (
            labels[sibling_index] == "ATT"
            and sibling_head >= 0
            and is_content_head(
                words[sibling_head],
                pos_tags[sibling_head],
                segmentation_words,
            )
        ):
            target = sibling_head
        else:
            target = resolve_modifier_target(
                sibling_index,
                words,
                pos_tags,
                heads,
                labels,
                segmentation_words,
            )
        if target is not None and not contains_boundary(
            modifier_index,
            target,
            words,
            pos_tags,
            include_de=True,
        ):
            record_repair(
                modifier_index,
                target,
                "coordination",
                "high",
                "并列修饰语继承同组修饰语的ATT中心词",
                label,
            )

    # 对反向ATT，优先寻找右侧同一短语中的中心词。
    for item in direct_att:
        modifier_index = int(item["modifier_index"])
        head_index = int(item["head_index"])
        if modifier_index <= head_index:
            continue
        target = resolve_modifier_target(
            modifier_index,
            words,
            pos_tags,
            heads,
            labels,
            segmentation_words,
            allow_verb_child=True,
        )
        if target is None:
            target = right_nominal_head_before_boundary(
                modifier_index,
                words,
                pos_tags,
                segmentation_words,
            )
        if target is not None:
            record_repair(
                modifier_index,
                target,
                "backward_att_repair",
                "high",
                "原ATT方向异常，重挂到右侧同一短语的内容中心词",
                "ATT",
            )

    final_relations = sorted(
        (
            relation
            for key, relation in relations.items()
            if key not in replaced_direct_keys
        ),
        key=lambda item: (
            int(item["modifier_index"]),
            int(item["head_index"]),
        ),
    )
    repairs.sort(
        key=lambda item: (
            int(item["modifier_index"]),
            int(item["head_index"]),
        )
    )
    return anomalies, repairs, final_relations


def analyze_sentence(
    sentence: str,
    words: list[str],
    pos_tags: list[str],
    dependency: dict[str, list[object]],
    semantic_dependency: dict[str, list[object]],
    srl_frames: list[dict[str, Any]],
    segmentation_words: set[str],
) -> dict[str, object]:
    words = [clean_token(word) for word in words]
    pos_tags = [clean_token(pos) for pos in pos_tags]
    heads = [int(head) for head in dependency["head"]]
    labels = [clean_token(label) for label in dependency["label"]]
    sdp_heads = [int(head) for head in semantic_dependency["head"]]
    sdp_labels = [
        clean_token(label)
        for label in semantic_dependency["label"]
    ]

    if not (
        len(words)
        == len(pos_tags)
        == len(heads)
        == len(labels)
        == len(sdp_heads)
        == len(sdp_labels)
    ):
        raise ValueError(
            "LTP结果长度不一致："
            f"cws={len(words)}, pos={len(pos_tags)}, "
            f"dep_heads={len(heads)}, dep_labels={len(labels)}, "
            f"sdp_heads={len(sdp_heads)}, sdp_labels={len(sdp_labels)}"
        )

    spans = locate_token_spans(sentence, words)
    srl_by_predicate = normalize_srl_frames(srl_frames)
    candidates: list[dict[str, object]] = []
    raw_att: list[dict[str, object]] = []
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
        decision, reason, keep = classify_atomic_att(
            item["modifier"],
            item["modifier_pos"],
            item["head"],
            item["head_pos"],
            segmentation_words,
        )
        raw_att.append(
            {
                **item,
                "decision": decision,
                "reason": reason,
                "keep": keep,
            }
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

    anomalies, repair_candidates, repaired_atomic_att = (
        build_atomic_att_repairs(
            words,
            pos_tags,
            heads,
            labels,
            final_atomic_att,
            srl_by_predicate,
            segmentation_words,
            sdp_heads,
            sdp_labels,
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
        "words": words,
        "pos_tags": pos_tags,
        "token_spans": spans,
        "segmentation": " / ".join(words),
        "word_pos": "；".join(
            f"{word}/{pos}"
            for word, pos in zip(words, pos_tags, strict=True)
        ),
        "raw_att": raw_att,
        "final_atomic_att": final_atomic_att,
        "atomic_anomalies": anomalies,
        "atomic_repair_candidates": repair_candidates,
        "repaired_atomic_att": repaired_atomic_att,
        "candidates": candidates,
        "complete_att": complete_att,
    }


def format_atomic(candidates: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']}/{item['modifier_pos']} → "
        f"{item['head']}/{item['head_pos']}"
        for item in candidates
    ) or "无"


def format_raw_att(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']}/{item['modifier_pos']} → "
        f"{item['head']}/{item['head_pos']}"
        f"（{item['decision']}：{item['reason']}）"
        for item in items
    ) or "无"


def format_atomic_anomalies(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']}/{item['modifier_pos']} → "
        f"{item['head']}/{item['head_pos']}"
        f"（{item['anomaly']}：{item['reason']}）"
        for item in items
    ) or "无"


def format_atomic_repairs(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']} → {item['head']}"
        f"（{item['source']}/{item['confidence']}：{item['reason']}）"
        for item in items
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
    dimension_lines = args.dimension_input.read_text(
        encoding="utf-8"
    ).splitlines()
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
    merge_rules = load_merge_rules_module()

    analyses: list[dict[str, object]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        result = model.pipeline(
            [sentence for _, sentence in batch],
            tasks=["cws", "pos", "dep", "sdp", "srl"],
        )
        for (
            (line_number, sentence),
            words,
            pos_tags,
            dep,
            sdp,
            srl,
        ) in zip(
            batch,
            result.cws,
            result.pos,
            result.dep,
            result.sdp,
            result.srl,
            strict=True,
        ):
            analysis = analyze_sentence(
                sentence,
                words,
                pos_tags,
                dep,
                sdp,
                srl,
                segmentation_word_set,
            )
            dimension_index = line_number - 1
            dimension_missing = dimension_index >= len(dimension_lines)
            dimension_sentence = (
                ""
                if dimension_missing
                else dimension_lines[dimension_index].strip()
            )
            dimension_alignment = align_excluded_dimension_positions(
                sentence,
                dimension_sentence,
                missing_as_empty=dimension_missing,
            )
            (
                dimension_filtered_atomic_att,
                dimension_removed_atomic_att,
            ) = filter_atomic_relations_by_dimensions(
                analysis["repaired_atomic_att"],
                analysis["token_spans"],
                dimension_alignment["excluded_positions"],
            )
            graph_merge_candidates = merge_rules.graph_merge_candidates(
                sentence,
                analysis["words"],
                analysis["pos_tags"],
                analysis["token_spans"],
                dimension_filtered_atomic_att,
                dimension_alignment["excluded_positions"],
            )
            srl_merge_candidates = merge_rules.srl_merge_candidates(
                sentence,
                analysis["words"],
                analysis["pos_tags"],
                analysis["token_spans"],
                dimension_filtered_atomic_att,
                analysis["candidates"],
                dimension_alignment["excluded_positions"],
            )
            compact_entity_candidates = (
                merge_rules.compact_entity_candidates(
                    sentence,
                    analysis["words"],
                    analysis["pos_tags"],
                    analysis["token_spans"],
                    dimension_filtered_atomic_att,
                    dimension_alignment["excluded_positions"],
                )
            )
            structural_surface_candidates = (
                merge_rules.structural_surface_candidates(
                    sentence,
                    analysis["words"],
                    analysis["pos_tags"],
                    analysis["token_spans"],
                    dimension_filtered_atomic_att,
                    dimension_alignment["excluded_positions"],
                )
            )
            rule_merged_att = merge_rules.select_merged_results(
                graph_merge_candidates,
                srl_merge_candidates,
                compact_entity_candidates,
                METRIC_HEAD_PATTERN,
                structural_surface_candidates,
            )
            analysis.update(
                {
                    "source_line": line_number,
                    "sentence": sentence,
                    **dimension_alignment,
                    "dimension_filtered_atomic_att": (
                        dimension_filtered_atomic_att
                    ),
                    "dimension_removed_atomic_att": (
                        dimension_removed_atomic_att
                    ),
                    "rule_merged_att": rule_merged_att,
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
    raw_att_count = sum(
        len(analysis["raw_att"])
        for analysis in analyses
    )
    pos_filtered_relation_count = sum(
        len(analysis["final_atomic_att"])
        for analysis in analyses
    )
    anomaly_count = sum(
        len(analysis["atomic_anomalies"])
        for analysis in analyses
    )
    repair_count = sum(
        len(analysis["atomic_repair_candidates"])
        for analysis in analyses
    )
    repaired_relation_count = sum(
        len(analysis["repaired_atomic_att"])
        for analysis in analyses
    )
    dimension_filtered_relation_count = sum(
        len(analysis["dimension_filtered_atomic_att"])
        for analysis in analyses
    )
    dimension_removed_relation_count = sum(
        len(analysis["dimension_removed_atomic_att"])
        for analysis in analyses
    )
    rule_merged_relation_count = sum(
        len(analysis["rule_merged_att"])
        for analysis in analyses
    )
    rule_merged_sentence_count = sum(
        bool(analysis["rule_merged_att"])
        for analysis in analyses
    )

    output_lines = [
        f"# {args.input.stem} 原子ATT纠错与补召回",
        "",
        f"- 来源：`{args.input.as_posix()}`",
        f"- 模型：`{args.model}`",
        "- LTP任务：`cws + pos + dep + sdp + srl`。",
        f"- 自定义词典：`{display_project_path(args.segmentation_dictionary)}`，"
        f"共{len(segmentation_words)}词。",
        f"- 自定义词频：`{args.segmentation_word_frequency}`。",
        "- 主任务：保留可信DEP-ATT，并结合SDP、局部词序和依存路径"
        "修复漏边、错挂及不可丢失的词法修饰。",
        "- SDP作用：只接收受约束的FEAT/dFEAT/mNEG/MANN等证据，"
        "不直接照搬全部语义依存边。",
        "- SRL作用：只提供动词谓词和目标论元证据，不再替换原子ATT。",
        "- 修复约束：原子ATT只连接原句中的现有token，不补词、"
        "不调序、不重建完整定语。",
        f"- 维度后问题：`{display_project_path(args.dimension_input)}`。",
        "- 维度过滤：与原问题按物理行号进行纯删除对齐；修饰词或"
        "中心词命中已删除维度时过滤整条原子关系；空行及缺失行"
        "按空结果处理。",
        "- 规则合并：最后一列复用第4.1阶段的无LLM规则；不修改前面的"
        "原子ATT结果；允许恢复原句中显式的后置状态/原因结构，"
        "但不改写、不补词、不做Schema映射。",
        f"- 原子ATT统计：DEP原始{raw_att_count}条，"
        f"第四阶段POS筛选后{pos_filtered_relation_count}条，"
        f"发现结构异常{anomaly_count}条，"
        f"生成修复候选{repair_count}条，"
        f"第四阶段最终{repaired_relation_count}条，"
        f"去除维度后{dimension_filtered_relation_count}条"
        f"（过滤{dimension_removed_relation_count}条）。",
        f"- 规则合并统计：{rule_merged_sentence_count}句产生"
        f"{rule_merged_relation_count}条结果。",
        f"- SRL旁路统计：动词ATT候选{len(all_candidates)}条，"
        f"连续片段候选{candidate_span_count}条，"
        f"原规则接受{accepted_recovery_count}条；"
        "这些片段只作后续证据，不进入第四阶段原子ATT。",
        "",
        "| 原文件行号 | 原句 | 分词（cws） | 词性（pos） | "
        "DEP原始ATT及POS判定 | POS筛选ATT | 异常ATT | "
        "DEP动词ATT | SRL证据 | 恢复判定 | SRL连续候选 | "
        "接受的动词定语 | 原子ATT修复候选 | 第四阶段原子ATT | "
        "第四阶段去除维度原子ATT | 规则合并结果 |",
        "|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for analysis in analyses:
        candidates = analysis["candidates"]
        output_lines.append(
            f"| {analysis['source_line']} | "
            f"{escape_table_cell(analysis['sentence'])} | "
            f"{escape_table_cell(analysis['segmentation'])} | "
            f"{escape_table_cell(analysis['word_pos'])} | "
            f"{escape_table_cell(format_raw_att(analysis['raw_att']))} | "
            f"{escape_table_cell(format_plain_att(analysis['final_atomic_att']))} | "
            f"{escape_table_cell(format_atomic_anomalies(analysis['atomic_anomalies']))} | "
            f"{escape_table_cell(format_atomic(candidates))} | "
            f"{escape_table_cell(format_srl_evidence(candidates))} | "
            f"{escape_table_cell(format_decisions(candidates))} | "
            f"{escape_table_cell(format_candidate_spans(candidates))} | "
            f"{escape_table_cell(format_recovered(candidates))} |"
            f" {escape_table_cell(format_atomic_repairs(analysis['atomic_repair_candidates']))} | "
            f"{escape_table_cell(format_plain_att(analysis['repaired_atomic_att']))} | "
            f"{escape_table_cell(format_plain_att(analysis['dimension_filtered_atomic_att']))} | "
            f"{escape_table_cell(merge_rules.format_merged_or_atomic(analysis['rule_merged_att'], analysis['dimension_filtered_atomic_att']))} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        f"已写入 {args.output}：{len(analyses)}句；"
        f"DEP原始ATT{raw_att_count}条，"
        f"修复候选{repair_count}条，"
        f"第四阶段原子ATT{repaired_relation_count}条；"
        f"去除维度后{dimension_filtered_relation_count}条；"
        f"规则合并{rule_merged_relation_count}条；"
        f"SRL连续候选{candidate_span_count}条"
    )


if __name__ == "__main__":
    main()
