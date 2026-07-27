from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

from ltp import LTP


# 第2阶段定稿词典：只保留需要干预分词的中文业务词。
SEGMENTATION_WORDS: list[str] = [
    "运营商BG",
    "中东中亚",
    "南部非洲",
    "北部非洲",
    "印度尼西亚",
    "中国区",
    "地区部",
    "代表处",
    "系统部",
    "亚太",
    "拉美",
    "南亚",
    "印尼",
    "二级以上",
    "一二级",
    "高风险",
    "中风险",
    "S级",
    "A级",
    "一级",
    "二级",
    "高危",
    "重大",
    "供方类问题",
    "产品质量",
    "服务质量",
    "网络质量",
    "物料供应",
    "分包资源",
    "供方问题",
    "解决方案",
    "华为问题",
    "设备增量",
    "国家三领先",
    "三领先",
    "领导力践行",
    "交付售前",
    "管理升级",
    "网上事故",
    "变更倒回",
    "变更操作",
    "疲劳驾驶",
    "小国小网",
    "网络维护",
    "客户声音",
    "工程实施",
    "比拼网络",
    "比拼项目",
    "系统集成",
    "数据中心",
    "网络规划",
    "工程优化",
    "数据分析",
    "辅助运营",
    "网络部署",
    "提示单",
    "客满",
    "完成率",
    "成功率",
    "及时率",
    "成本率",
    "利润率",
    "销毛率",
    "成熟度",
    "占比",
    "根因",
]

NOUN_LIKE_POS = {"n", "nl", "ns", "nt", "nz", "ni", "nh", "ws"}
ATTRIBUTE_POS = {"a", "b", "j", "nd"}
QUESTION_OR_QUANTITY_POS = {"r", "m", "q"}
RECOVERABLE_ATTRIBUTE_WORDS = {
    "二级以上",
    "一二级",
    "高风险",
    "中风险",
    "S级",
    "A级",
    "一级",
    "二级",
    "高危",
    "重大",
    "交付",
    "网络质量",
}
LOCAL_ENTITY_HEADS = {
    "项目",
    "客户",
    "事故",
    "网上事故",
    "管理升级",
    "变更操作",
    "变更倒回",
    "比拼网络",
    "比拼项目",
    "网络",
    "提示单",
}
CONTENT_CHILD_LABELS = {
    "ATT",
    "ADV",
    "SBV",
    "VOB",
    "FOB",
    "IOB",
    "POB",
    "CMP",
    "COO",
    "LAD",
}
CLAUSE_MARKERS = {"由于", "由"}
SCOPE_BOUNDARIES = {
    "全球",
    "运营商",
    "业务",
    "运营商BG",
    "亚太",
    "拉美",
    "中东中亚",
    "南部非洲",
    "北部非洲",
    "南亚",
    "欧洲",
    "中国区",
    "印尼",
    "印度尼西亚",
    "泰国",
    "地区部",
    "代表处",
    "系统部",
}

PUNCTUATION_PATTERN = re.compile(r"^[，,。！？!?；;：:、]$")
ASCII_TERM_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用自定义词典、POS 和 DEP 提取完整定语。"
    )
    parser.add_argument("input", type=Path, help="每行一句的 Markdown 文本")
    parser.add_argument("output", type=Path, help="Markdown 输出文件")
    parser.add_argument(
        "--dimension-input",
        type=Path,
        help=(
            "可选：与原文件按物理行号对齐的维度提取后问题；"
            "缺失行按空结果处理"
        ),
    )
    parser.add_argument("--model", default="LTP/base", help="LTP 模型名称或路径")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--segmentation-word-frequency",
        type=int,
        default=2,
        help="自定义词典分词权重（默认 2）",
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


def locate_token_spans(sentence: str, words: list[str]) -> list[tuple[int, int]]:
    """将 LTP token 对齐回原句，区间采用左闭右开。"""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        start = sentence.find(word, cursor)
        if start < 0:
            raise ValueError(
                f"token 无法对齐原句：token={word!r}, cursor={cursor}, "
                f"sentence={sentence!r}"
            )
        end = start + len(word)
        spans.append((start, end))
        cursor = end
    return spans


def normalize_with_positions(text: str) -> tuple[str, list[int]]:
    """移除空白并保留规范化字符到原句字符位置的映射。"""
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char)
        positions.append(index)
    return "".join(chars), positions


def merge_character_positions(
    positions: set[int],
    sentence: str,
) -> list[tuple[int, int]]:
    """把离散字符位置合并成原句中的左闭右开区间。"""
    if not positions:
        return []
    ordered = sorted(positions)
    spans: list[tuple[int, int]] = []
    start = ordered[0]
    previous = ordered[0]
    for position in ordered[1:]:
        between = sentence[previous + 1 : position]
        if position == previous + 1 or not between.strip():
            previous = position
            continue
        spans.append((start, previous + 1))
        start = position
        previous = position
    spans.append((start, previous + 1))
    return spans


def align_dimension_sentence(
    original: str,
    dimension_sentence: str,
    *,
    missing_as_empty: bool,
) -> dict[str, object]:
    """定位原句中被维度提取过程删除的字符，不进行字符串改写。"""
    normalized_original, original_positions = normalize_with_positions(original)
    normalized_dimension, _ = normalize_with_positions(dimension_sentence)

    if not normalized_dimension:
        excluded_positions = {
            position
            for position in original_positions
            if not original[position].isspace()
        }
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
        for tag, original_start, original_end, dim_start, dim_end in matcher.get_opcodes():
            if tag == "equal":
                retained_normalized.update(range(original_start, original_end))
            elif tag != "delete":
                invalid_operations.append(
                    f"{tag}:{original_start}-{original_end}/"
                    f"{dim_start}-{dim_end}"
                )
        if invalid_operations:
            raise ValueError(
                "维度后问题不是原句的纯删除结果："
                f"original={original!r}, dimension={dimension_sentence!r}, "
                f"operations={invalid_operations}"
            )

        excluded_positions = {
            original_positions[index]
            for index in range(len(original_positions))
            if index not in retained_normalized
        }
        status = "未删除维度" if not excluded_positions else "已对齐"

    excluded_spans = merge_character_positions(excluded_positions, original)
    excluded_texts = [
        original[start:end].strip()
        for start, end in excluded_spans
        if original[start:end].strip()
    ]
    return {
        "dimension_sentence": dimension_sentence,
        "dimension_status": status,
        "excluded_positions": excluded_positions,
        "excluded_spans": excluded_spans,
        "excluded_texts": excluded_texts,
    }


def filter_candidates_by_dimensions(
    candidates: list[dict[str, object]],
    excluded_positions: set[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """定语或中心词字符区间碰到已删除维度，就过滤整条候选。"""
    kept: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for candidate in candidates:
        start = int(candidate["start"])
        end = int(candidate["end"])
        head_start = int(candidate["head_start"])
        head_end = int(candidate["head_end"])
        modifier_overlap = sorted(
            position
            for position in excluded_positions
            if start <= position < end
        )
        head_overlap = sorted(
            position
            for position in excluded_positions
            if head_start <= position < head_end
        )
        keep = not modifier_overlap and not head_overlap
        if modifier_overlap and head_overlap:
            filter_reason = "定语和中心词均包含已抽取维度"
        elif modifier_overlap:
            filter_reason = "定语包含已抽取维度"
        else:
            filter_reason = "中心词包含已抽取维度"
        decisions.append(
            {
                "action": "保留" if keep else "过滤",
                "reason": (
                    "定语和中心词均未包含已抽取维度"
                    if keep
                    else filter_reason
                ),
                "modifier_text": candidate["modifier_text"],
                "head_text": candidate["head_text"],
            }
        )
        if keep:
            kept.append(candidate)
    return kept, decisions


def build_children(heads: list[int]) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in heads]
    for child, head in enumerate(heads):
        if head > 0:
            children[head - 1].append(child)
    return children


def descendants(root: int, children: list[list[int]]) -> set[int]:
    found: set[int] = set()
    stack = list(children[root])
    while stack:
        index = stack.pop()
        if index in found:
            continue
        found.add(index)
        stack.extend(children[index])
    return found


def is_time_word(word: str, pos: str) -> bool:
    return pos == "nt" or TIME_PATTERN.fullmatch(word) is not None


def has_rad_de(
    modifier: int,
    words: list[str],
    labels: list[str],
    children: list[list[int]],
) -> bool:
    return any(
        labels[child] == "RAD"
        and words[child] == "的"
        and child > modifier
        for child in children[modifier]
    )


def classify_att(
    modifier: int,
    head_index: int,
    words: list[str],
    pos_tags: list[str],
    labels: list[str],
    children: list[list[int]],
) -> tuple[str, str, bool]:
    """返回（动作、原因、是否保留）。POS 只辅助分类，不硬删业务词。"""
    word = words[modifier]
    pos = pos_tags[modifier]
    head = words[head_index]

    if has_rad_de(modifier, words, labels, children):
        if pos.startswith("v"):
            return "重建", "动词性定语从句", True
        return "重建", "带“的”定语短语", True

    if (
        YEAR_PATTERN.fullmatch(word) is not None
        and MONTH_PATTERN.fullmatch(head) is not None
    ):
        return "过滤", "日历内部修饰", False

    if word in SEGMENTATION_WORDS:
        return "保留", "业务词（POS覆盖）", True

    if PUNCTUATION_PATTERN.fullmatch(word) is not None or pos in {"wp", "u"}:
        return "过滤", "标点或结构助词", False

    if NOISE_WORD_PATTERN.fullmatch(word) is not None:
        return "过滤", "疑问、指示或数量噪声", False

    if pos in QUESTION_OR_QUANTITY_POS:
        return "过滤", f"低价值词性/{pos}", False

    if is_time_word(word, pos):
        return "过滤", "时间修饰", False

    if pos in NOUN_LIKE_POS:
        return "保留", f"名词性/{pos}", True

    if pos in ATTRIBUTE_POS:
        return "保留", f"属性性/{pos}", True

    if pos.startswith("v"):
        # 中文业务事件经常被标成动词；处于 ATT 位置时保留，
        # 但如果带“的”已经在上面按完整定语从句处理。
        return "保留", f"动词性修饰/{pos}", True

    return "保留", f"其他内容词/{pos}", True


def resolve_content_head(
    head_index: int,
    words: list[str],
    pos_tags: list[str],
    heads: list[int],
) -> int:
    """当 ATT 错挂到量词/疑问词时，沿依存中心提升到内容词。"""
    original = head_index
    current = head_index
    visited: set[int] = set()
    while current not in visited:
        visited.add(current)
        word = words[current]
        pos = pos_tags[current]
        if (
            word not in SEGMENTATION_WORDS
            and (
                NOISE_WORD_PATTERN.fullmatch(word) is not None
                or pos in QUESTION_OR_QUANTITY_POS
            )
            and heads[current] > 0
        ):
            current = heads[current] - 1
            continue
        break
    if current == original:
        return original
    if pos_tags[current] in NOUN_LIKE_POS or words[current] in LOCAL_ENTITY_HEADS:
        return current
    return original


def find_local_entity_head(
    modifier: int,
    words: list[str],
) -> int | None:
    """为等级、风险和“交付”等漏标词寻找右侧近邻业务实体。"""
    upper = min(len(words), modifier + 5)
    for index in range(modifier + 1, upper):
        if PUNCTUATION_PATTERN.fullmatch(words[index]) is not None:
            break
        if words[index] in LOCAL_ENTITY_HEADS:
            return index
    return None


def is_noise_head(index: int, words: list[str], pos_tags: list[str]) -> bool:
    """词典业务词优先于 LTP 的 m/q/r 词性判断。"""
    return words[index] not in SEGMENTATION_WORDS and (
        NOISE_WORD_PATTERN.fullmatch(words[index]) is not None
        or pos_tags[index] in QUESTION_OR_QUANTITY_POS
    )


def is_branch_boundary(word: str, pos: str) -> bool:
    return (
        not word
        or word in SCOPE_BOUNDARIES
        or PUNCTUATION_PATTERN.fullmatch(word) is not None
        or NOISE_WORD_PATTERN.fullmatch(word) is not None
        or is_time_word(word, pos)
    )


def collect_content_branch(
    root: int,
    modifier: int,
    words: list[str],
    pos_tags: list[str],
    labels: list[str],
    children: list[list[int]],
) -> set[int]:
    selected: set[int] = set()
    stack = [root]
    while stack:
        index = stack.pop()
        if index > modifier or is_branch_boundary(words[index], pos_tags[index]):
            continue
        selected.add(index)
        for child in children[index]:
            if child <= modifier and labels[child] in CONTENT_CHILD_LABELS:
                stack.append(child)
    return selected


def reconstruct_relative_clause(
    sentence: str,
    modifier: int,
    words: list[str],
    pos_tags: list[str],
    labels: list[str],
    spans: list[tuple[int, int]],
    children: list[list[int]],
) -> tuple[str, int, int] | None:
    rad_children = sorted(
        child
        for child in children[modifier]
        if labels[child] == "RAD" and words[child] == "的" and child > modifier
    )
    if not rad_children:
        return None

    subtree = descendants(modifier, children)
    end_token = rad_children[0]
    markers = sorted(
        index
        for index in subtree
        if index < modifier and words[index] in CLAUSE_MARKERS
    )

    if markers:
        start_token = markers[-1]
    else:
        selected = {modifier}
        for child in children[modifier]:
            if (
                child < modifier
                and labels[child] in CONTENT_CHILD_LABELS
                and not is_branch_boundary(words[child], pos_tags[child])
            ):
                selected.update(
                    collect_content_branch(
                        child,
                        modifier,
                        words,
                        pos_tags,
                        labels,
                        children,
                    )
                )
        start_token = min(selected)

    punctuation = [
        index
        for index in range(start_token, modifier)
        if PUNCTUATION_PATTERN.fullmatch(words[index]) is not None
    ]
    if punctuation:
        start_token = punctuation[-1] + 1

    start_char = spans[start_token][0]
    end_char = spans[end_token][1]
    text = sentence[start_char:end_char].strip()
    if not text:
        return None
    return text, start_char, end_char


def find_copular_clause_candidates(
    sentence: str,
    words: list[str],
    pos_tags: list[str],
    heads: list[int],
    labels: list[str],
    spans: list[tuple[int, int]],
    children: list[list[int]],
) -> list[dict[str, object]]:
    """补充“项目是由于……导致的”一类不被标成 ATT 的后置条件。"""
    candidates: list[dict[str, object]] = []
    for core, (head, label) in enumerate(zip(heads, labels, strict=True)):
        if label == "ATT" or head == 0:
            continue
        reconstructed = reconstruct_relative_clause(
            sentence,
            core,
            words,
            pos_tags,
            labels,
            spans,
            children,
        )
        if reconstructed is None:
            continue

        subtree = descendants(core, children)
        markers = [
            index
            for index in subtree
            if index < core and words[index] in CLAUSE_MARKERS
        ]
        copula = head - 1
        if not markers or words[copula] != "是":
            continue

        targets = [
            index
            for index in children[copula]
            if index < copula and labels[index] in {"SBV", "DBL", "FOB", "VOB"}
        ]
        if not targets and heads[copula] > 0:
            targets = [
                index
                for index, sibling_head in enumerate(heads)
                if index < copula
                and sibling_head == heads[copula]
                and labels[index] in {"SBV", "DBL", "FOB", "VOB"}
            ]
        if not targets:
            continue

        modifier_text, start_char, end_char = reconstructed
        target = max(targets)
        candidates.append(
            {
                "modifier_text": modifier_text,
                "head_text": words[target],
                "modifier_core": words[core],
                "modifier_pos": pos_tags[core],
                "head_pos": pos_tags[target],
                "source": "系词后置定语从句",
                "start": start_char,
                "end": end_char,
                "head_index": target,
                "head_start": spans[target][0],
                "head_end": spans[target][1],
            }
        )
    return candidates


def analyze_sentence(
    sentence: str,
    words: list[str],
    pos_tags: list[str],
    dependency: dict[str, list[object]],
) -> dict[str, object]:
    words = [clean_token(word) for word in words]
    pos_tags = [clean_token(pos) for pos in pos_tags]
    heads = [int(head) for head in dependency["head"]]
    labels = [clean_token(label) for label in dependency["label"]]
    spans = locate_token_spans(sentence, words)
    children = build_children(heads)

    raw_att: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    final_candidates: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    raw_att_modifiers: set[int] = set()

    for modifier, (head, label) in enumerate(zip(heads, labels, strict=True)):
        if label != "ATT" or head == 0:
            continue
        raw_att_modifiers.add(modifier)
        head_index = head - 1
        raw_att.append(
            {
                "modifier_text": words[modifier],
                "modifier_pos": pos_tags[modifier],
                "head_text": words[head_index],
                "head_pos": pos_tags[head_index],
            }
        )

        action, reason, keep = classify_att(
            modifier,
            head_index,
            words,
            pos_tags,
            labels,
            children,
        )
        decisions.append(
            {
                "action": action,
                "reason": reason,
                "modifier_text": words[modifier],
                "modifier_pos": pos_tags[modifier],
                "head_text": words[head_index],
                "head_pos": pos_tags[head_index],
            }
        )
        if not keep:
            continue

        local_target = (
            find_local_entity_head(modifier, words)
            if words[modifier] in RECOVERABLE_ATTRIBUTE_WORDS
            and is_noise_head(head_index, words, pos_tags)
            else None
        )
        final_head_index = (
            local_target
            if local_target is not None
            else resolve_content_head(
                head_index,
                words,
                pos_tags,
                heads,
            )
        )
        if final_head_index != head_index:
            decisions[-1]["promoted_head_text"] = words[final_head_index]
            decisions[-1]["promoted_head_pos"] = pos_tags[final_head_index]
        elif is_noise_head(head_index, words, pos_tags):
            decisions[-1]["action"] = "过滤"
            decisions[-1]["reason"] = "无法提升的数量中心词"
            continue

        if action == "重建":
            reconstructed = reconstruct_relative_clause(
                sentence,
                modifier,
                words,
                pos_tags,
                labels,
                spans,
                children,
            )
            if reconstructed is None:
                continue
            modifier_text, start_char, end_char = reconstructed
            source = "定语从句"
        else:
            modifier_text = words[modifier]
            start_char, end_char = spans[modifier]
            source = reason

        key = (modifier_text, final_head_index)
        if key in seen:
            continue
        seen.add(key)
        final_candidates.append(
            {
                "modifier_text": modifier_text,
                "head_text": words[final_head_index],
                "modifier_core": words[modifier],
                "modifier_pos": pos_tags[modifier],
                "head_pos": pos_tags[final_head_index],
                "source": source,
                "start": start_char,
                "end": end_char,
                "head_index": final_head_index,
                "head_start": spans[final_head_index][0],
                "head_end": spans[final_head_index][1],
            }
        )

    # LTP 偶尔把“中风险”标为 RAD，或把“交付”分析成主句谓词。
    # 只对有限业务属性和连续英文数字专名，在右侧四个 token 内
    # 寻找明确业务实体。
    for modifier, word in enumerate(words):
        if modifier in raw_att_modifiers or (
            word not in RECOVERABLE_ATTRIBUTE_WORDS
            and ASCII_TERM_PATTERN.fullmatch(word) is None
        ):
            continue
        target = find_local_entity_head(modifier, words)
        if target is None:
            continue
        key = (word, target)
        if key in seen:
            continue
        seen.add(key)
        decisions.append(
            {
                "action": "补充",
                "reason": "POS+局部名词短语恢复",
                "modifier_text": word,
                "modifier_pos": pos_tags[modifier],
                "head_text": words[target],
                "head_pos": pos_tags[target],
            }
        )
        final_candidates.append(
            {
                "modifier_text": word,
                "head_text": words[target],
                "modifier_core": word,
                "modifier_pos": pos_tags[modifier],
                "head_pos": pos_tags[target],
                "source": "POS+局部名词短语恢复",
                "start": spans[modifier][0],
                "end": spans[modifier][1],
                "head_index": target,
                "head_start": spans[target][0],
                "head_end": spans[target][1],
            }
        )

    for candidate in find_copular_clause_candidates(
        sentence,
        words,
        pos_tags,
        heads,
        labels,
        spans,
        children,
    ):
        key = (str(candidate["modifier_text"]), int(candidate["head_index"]))
        if key not in seen:
            seen.add(key)
            final_candidates.append(candidate)

    final_candidates.sort(
        key=lambda item: (
            int(item["head_index"]),
            int(item["start"]),
            int(item["end"]),
        )
    )

    dependency_text = "；".join(
        f"{index + 1}:{word} -[{labels[index]}]→ "
        f"{'0:ROOT' if heads[index] == 0 else f'{heads[index]}:{words[heads[index] - 1]}'}"
        for index, word in enumerate(words)
    )

    return {
        "segmentation": " / ".join(words),
        "word_pos": "；".join(
            f"{word}/{pos}" for word, pos in zip(words, pos_tags, strict=True)
        ),
        "dependency": dependency_text,
        "raw_att": raw_att,
        "decisions": decisions,
        "final_candidates": final_candidates,
    }


def format_raw_att(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier_text']}/{item['modifier_pos']} → "
        f"{item['head_text']}/{item['head_pos']}"
        for item in items
    ) or "无"


def format_decisions(items: list[dict[str, object]]) -> str:
    formatted: list[str] = []
    for item in items:
        text = (
            f"{item['action']}[{item['reason']}]："
            f"{item['modifier_text']}/{item['modifier_pos']} → "
            f"{item['head_text']}/{item['head_pos']}"
        )
        if "promoted_head_text" in item:
            text += (
                f" ⇒ {item['promoted_head_text']}/"
                f"{item['promoted_head_pos']}"
            )
        formatted.append(text)
    return "；".join(formatted) or "无"


def format_final(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier_text']} → {item['head_text']}"
        for item in items
    ) or "无"


def format_dimension_decisions(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['action']}[{item['reason']}]："
        f"{item['modifier_text']} → {item['head_text']}"
        for item in items
    ) or "无"


def main() -> None:
    args = parse_args()
    source_lines = args.input.read_text(encoding="utf-8").splitlines()
    dimension_lines = (
        args.dimension_input.read_text(encoding="utf-8").splitlines()
        if args.dimension_input
        else []
    )
    records = [
        (line_number, line.strip())
        for line_number, line in enumerate(source_lines, start=1)
        if line_number > 1 and line.strip()
    ]

    model = LTP(args.model)
    model.add_words(
        SEGMENTATION_WORDS,
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
            analysis = analyze_sentence(sentence, words, pos_tags, dep)
            analysis.update(
                {
                    "source_line": line_number,
                    "sentence": sentence,
                }
            )
            if args.dimension_input:
                dimension_index = line_number - 1
                missing_as_empty = dimension_index >= len(dimension_lines)
                dimension_sentence = (
                    ""
                    if missing_as_empty
                    else dimension_lines[dimension_index].strip()
                )
                dimension_alignment = align_dimension_sentence(
                    sentence,
                    dimension_sentence,
                    missing_as_empty=missing_as_empty,
                )
                filtered_candidates, dimension_decisions = (
                    filter_candidates_by_dimensions(
                        analysis["final_candidates"],
                        dimension_alignment["excluded_positions"],
                    )
                )
                analysis.update(dimension_alignment)
                analysis["dimension_decisions"] = dimension_decisions
                analysis["dimension_filtered_candidates"] = filtered_candidates
            else:
                analysis["dimension_filtered_candidates"] = analysis[
                    "final_candidates"
                ]
            analyses.append(analysis)

        completed = min(start + len(batch), len(records))
        print(f"已完成 {completed}/{len(records)}")

    if len(analyses) != len(records):
        raise RuntimeError(
            f"记录数不一致：输入 {len(records)}，输出 {len(analyses)}"
        )

    output_lines = [
        f"# {args.input.stem} POS 辅助 DEP 定语提取",
        "",
        f"- 来源：`{args.input.as_posix()}`",
        f"- 模型：`{args.model}`",
        "- LTP任务：`cws + pos + dep`。",
        f"- 自定义词典：`SEGMENTATION_WORDS`，共{len(SEGMENTATION_WORDS)}词。",
        f"- 自定义词频：`{args.segmentation_word_frequency}`。",
        "- POS作用：过滤疑问、数量和时间噪声；区分普通定语、业务词和定语从句。",
        "- 完整定语：只截取原句片段，不进行语义改写。",
        "- 当前阶段：不使用LLM、SRL、Schema映射或文本改写。",
        *(
            [
                f"- 维度后问题：`{args.dimension_input.as_posix()}`。",
                "- 维度过滤：按物理行号对齐；缺失行按空结果处理；"
                "定语或中心词字符区间与已删除维度重叠时过滤整条候选。",
            ]
            if args.dimension_input
            else ["- 维度过滤：未启用。"]
        ),
        "",
        *(
            [
                "| 原文件行号 | 原句 | 维度后问题 | 已删除维度片段 | "
                "分词（cws） | 词性（pos） | 原始ATT | POS辅助判定 | "
                "维度过滤前定语 | 维度过滤判定 | 最终定语（去维度） |",
                "|---:|---|---|---|---|---|---|---|---|---|---|",
            ]
            if args.dimension_input
            else [
                "| 原文件行号 | 原句 | 分词（cws） | 词性（pos） | "
                "原始ATT | POS辅助判定 | 最终完整定语 |",
                "|---:|---|---|---|---|---|---|",
            ]
        ),
    ]

    for analysis in analyses:
        if args.dimension_input:
            dimension_display = (
                analysis["dimension_sentence"]
                if analysis["dimension_sentence"]
                else f"（{analysis['dimension_status']}）"
            )
            excluded_display = "；".join(analysis["excluded_texts"]) or "无"
            output_lines.append(
                f"| {analysis['source_line']} | "
                f"{escape_table_cell(analysis['sentence'])} | "
                f"{escape_table_cell(dimension_display)} | "
                f"{escape_table_cell(excluded_display)} | "
                f"{escape_table_cell(analysis['segmentation'])} | "
                f"{escape_table_cell(analysis['word_pos'])} | "
                f"{escape_table_cell(format_raw_att(analysis['raw_att']))} | "
                f"{escape_table_cell(format_decisions(analysis['decisions']))} | "
                f"{escape_table_cell(format_final(analysis['final_candidates']))} | "
                f"{escape_table_cell(format_dimension_decisions(analysis['dimension_decisions']))} | "
                f"{escape_table_cell(format_final(analysis['dimension_filtered_candidates']))} |"
            )
        else:
            output_lines.append(
                f"| {analysis['source_line']} | "
                f"{escape_table_cell(analysis['sentence'])} | "
                f"{escape_table_cell(analysis['segmentation'])} | "
                f"{escape_table_cell(analysis['word_pos'])} | "
                f"{escape_table_cell(format_raw_att(analysis['raw_att']))} | "
                f"{escape_table_cell(format_decisions(analysis['decisions']))} | "
                f"{escape_table_cell(format_final(analysis['final_candidates']))} |"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        f"已写入 {args.output}：{len(analyses)} 句；"
        f"词典{len(SEGMENTATION_WORDS)}词，freq={args.segmentation_word_frequency}"
    )


if __name__ == "__main__":
    main()
