from __future__ import annotations

import argparse
import difflib
import re
from collections import Counter
from pathlib import Path

from reorder_with_hanlp import DEFAULT_INPUT, DEFAULT_MODEL_HOME, load_analyses
from reordering_core import (
    Decision,
    candidate_sentences,
    make_plan,
    read_records,
    validate_plan,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "original_question_reordered_dep_att.md"
DEFAULT_DIMENSION_INPUT = ROOT.parent / "data" / "dimension_extracted_question.md"
ATT_LABELS = {"nn", "amod", "assmod", "rcmod", "vmod"}
PREDICATE_MODIFIER_LABELS = {"advmod", "neg"}
NON_CORE_PREDICATE_MODIFIERS = {"还", "仍", "依然"}
CAUSE_MARKERS = {"由", "由于", "因为"}
RECOVERABLE_PREDICATES = {"导致", "造成", "引起", "产生", "存在", "有", "无", "关闭", "恢复"}
NOMINALIZED_COMPOUND_PREDICATES = {"比拼", "交付"}
RISK_LEVEL_PARTS = {"高", "中", "低"}
RISK_WORDS = {"高风险", "中风险", "低风险"}
COORDINATION_MARKERS = {"或", "和", "及"}
LOCATIVE_COMPOUND_SUFFIXES = {"上", "下", "中", "内", "外"}
ENTITY_CONDITION_WORDS = RISK_WORDS | {"高危", "低危", "重大"}
LEVEL_NUMBER_TOKENS = set("一二三四五六七八九十两")
LEVEL_CONNECTORS = {"和", "或", "及", "、"}
LEVEL_RANGE_SUFFIXES = {"以上", "以下"}
# 这些词既可能被POS标作动词，也稳定地作为业务复合名词的内部成分出现。
# 仅用于等级短语右侧的紧邻局部实体，不跨越谓词或介词。
NOMINAL_COMPOUND_PARTS = {"交付", "比拼", "升级", "倒回"}
# 从完整指标名尾部识别可独立理解的指标概念。按长度优先匹配，避免把
# “预算完成率”误截为“完成率”。左侧连续片段才作为被度量业务对象。
INDICATOR_TAILS = (
    "风险超期未关闭率",
    "预算完成率",
    "预测完成率",
    "恢复及时率",
    "超期未关闭率",
    "管理成熟度",
    "达成率",
    "成本率",
    "完成率",
    "成功率",
    "及时率",
    "成熟度",
    "占比",
    "总数",
    "次数",
    "数量",
    "单数",
    "项目数",
    "数",
)
DE_CLAUSE_TRIGGERS = {
    "相关", "保障", "合并", "达成", "领先", "比拼", "导致", "发生", "恢复",
    "新建", "搬迁", "健康", "不满",
}
GENERIC_OBJECT_PREFIXES = ("业务", "运营商业务")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从原始问题直接完成受控重排并输出HanLP DEP原子ATT关系。"
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dimension-input",
        type=Path,
        default=DEFAULT_DIMENSION_INPUT,
        help="与原问题逐行对应的维度提取结果；不存在时不做维度过滤",
    )
    parser.add_argument("--model-home", type=Path, default=DEFAULT_MODEL_HOME)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size必须大于0")
    return args


def escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def read_dimension_lines(path: Path) -> list[str]:
    """保留空行，以便与原始问题按行号对齐。"""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()[1:]]


def remove_time_from_dimension_question(value: str) -> str:
    """维度提取句仍保留时间；展示列中将时间作为独立条件移除。"""
    if not value:
        return "—"
    result = value
    patterns = [
        r"(?:截至|截止)\s*(?:当前|\d{4}年(?:\d{1,2}月)?)\s*[，,]?",
        r"\d{4}年(?:\d{1,2}月)?\s*[，,]?",
        r"\bH[12]\b\s*",
    ]
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    result = re.sub(r"^[，,\s]+|[，,\s]+$", "", result)
    result = re.sub(r"[，,]{2,}", "，", result)
    return result or "—"


def removed_dimension_words(original: str, dimension_sentence: str, words: list[str]) -> set[str]:
    """保守地取原句相对维度提取句被删除的 token 类型。

    下游当前按词文本过滤，无法区分同一词在不同位置的两个实例。因此只有
    一个词在原句中所有出现位置都落入删除范围时，才把它标为维度词；若该词
    还在其他位置保留，则宁可不自动过滤，避免误删业务实体。
    """
    if not dimension_sentence:
        return set()
    removed_ranges: list[tuple[int, int]] = []
    for tag, start, end, _other_start, _other_end in difflib.SequenceMatcher(
        None, original, dimension_sentence
    ).get_opcodes():
        if tag in {"delete", "replace"} and start < end:
            removed_ranges.append((start, end))
    occurrences = Counter(words)
    removed_occurrences: Counter[str] = Counter()
    cursor = 0
    for word in words:
        word_start, word_end = cursor, cursor + len(word)
        if any(word_start < end and word_end > start for start, end in removed_ranges):
            removed_occurrences[word] += 1
        cursor = word_end
    return {
        word
        for word, removed_count in removed_occurrences.items()
        if removed_count == occurrences[word]
    }


def dimensionless_final_sentence(
    analysis: object, dimension_words: set[str], has_dimension_sentence: bool
) -> str:
    """在最终重排句上移除已识别的维度词和时间表达。"""
    if not has_dimension_sentence:
        return "—"
    value = "".join(word for word in analysis.words if word not in dimension_words)
    return remove_time_from_dimension_question(value)


def is_time_token(word: str, pos: str) -> bool:
    """识别分词/POS偶尔未标为NT的时间片段。"""
    return pos == "NT" or bool(
        re.fullmatch(r"(?:\d{4}年(?:\d{1,2}月)?|\d{1,2}月|H[12]|当前)", word)
    )


def att_relations(analysis: object, dimension_words: set[str]) -> str:
    """输出已去除时间、维度后的DEP原子ATT关系。"""
    words = analysis.words
    predicate_modifiers: dict[int, list[tuple[int, str]]] = {}
    for index, (head, label) in enumerate(zip(analysis.heads, analysis.labels, strict=True)):
        if label in PREDICATE_MODIFIER_LABELS and words[index] not in NON_CORE_PREDICATE_MODIFIERS:
            predicate_modifiers.setdefault(head - 1, []).append((index, words[index]))

    relations: list[str] = []
    for index, (head, label) in enumerate(zip(analysis.heads, analysis.labels, strict=True)):
        if label not in ATT_LABELS or head == 0:
            continue
        head_index = head - 1
        # 时间关系与从原句/维度句差异中识别出的维度关系不进入原子ATT列。
        if (
            is_time_token(words[index], analysis.pos[index])
            or is_time_token(words[head_index], analysis.pos[head_index])
            or words[index] in dimension_words
            or words[head_index] in dimension_words
        ):
            continue
        dependent = words[index]
        if label in {"rcmod", "vmod"}:
            modifiers = "".join(
                word for _position, word in sorted(predicate_modifiers.get(index, []))
            )
            dependent = f"{modifiers}{dependent}"
        relations.append(f"{dependent} -[{label}]→ {words[head_index]}")
    return "；".join(relations) or "—"


def children(analysis: object, head_index: int, labels: set[str] | None = None) -> list[int]:
    return [
        index
        for index, (head, label) in enumerate(zip(analysis.heads, analysis.labels, strict=True))
        if head == head_index + 1 and (labels is None or label in labels)
    ]


def local_locative_compound_at(analysis: object, start: int) -> tuple[int, str] | None:
    """恢复被切开的局部“名词 + 上/下/内/外 + 名词”复合实体。

    仅接受连续三词及局部DEP链 ``N -lobj→ 方位词``，并要求右侧名词
    与方位词同指向一个中心词（或直接作为其中心词），所以不会把相隔较远
    的普通方位短语误并入实体。典型例子是
    ``网 / 上 / 事故``，恢复为“网上事故”。
    返回末尾中心词下标及恢复后的文本。
    """
    end = start + 2
    if end >= len(analysis.words):
        return None
    if (
        not analysis.pos[start].startswith("N")
        or analysis.words[start + 1] not in LOCATIVE_COMPOUND_SUFFIXES
        or not analysis.pos[end].startswith("N")
    ):
        return None
    left_to_suffix = analysis.heads[start] == start + 2
    suffix_to_right = analysis.heads[start + 1] == end + 1
    right_to_suffix = analysis.heads[end] == start + 2
    suffix_to_left = analysis.heads[start + 1] == start + 1
    same_rightward_head = (
        left_to_suffix
        and analysis.heads[start + 1] > end + 1
        and analysis.heads[end] == analysis.heads[start + 1]
    )
    is_forward_chain = left_to_suffix and suffix_to_right
    is_reverse_chain = right_to_suffix and suffix_to_left
    if not (is_forward_chain or is_reverse_chain or same_rightward_head):
        return None
    return end, "".join(analysis.words[start : end + 1])


def enrich_local_locative_compounds(analysis: object, selected: set[int]) -> set[int]:
    """若已选实体触及局部复合结构的任一端，则补齐整段词元。"""
    enriched = set(selected)
    for start in range(len(analysis.words) - 2):
        compound = local_locative_compound_at(analysis, start)
        if compound is None:
            continue
        end, _value = compound
        if start in selected or end in selected:
            enriched.update(range(start, end + 1))
    return enriched


def nominal_phrase(analysis: object, head_index: int, blocked_words: set[str] | None = None) -> str:
    """恢复名词及其局部名词/形容词修饰，不跨越谓词或介词结构。"""
    allowed = {"nn", "amod", "ordmod", "conj", "cc"}
    selected = {head_index}
    frontier = [head_index]
    while frontier:
        parent = frontier.pop()
        for child in children(analysis, parent, allowed):
            selected.add(child)
            frontier.append(child)
    blocked_words = blocked_words or set()
    return "".join(
        analysis.words[index] for index in sorted(selected) if analysis.words[index] not in blocked_words
    )


def static_subtree_indices(analysis: object, head_index: int) -> set[int]:
    """返回纯名词定中子树，用于从完整实体中剔除当前修饰语。"""
    allowed = {"nn", "amod", "assmod", "ordmod", "clf", "conj", "cc"}
    selected = {head_index}
    frontier = [head_index]
    while frontier:
        parent = frontier.pop()
        for child in children(analysis, parent, allowed):
            selected.add(child)
            frontier.append(child)
    return selected


def static_entity_without(
    analysis: object, head_index: int, excluded: set[int], blocked_words: set[str] | None = None
) -> str:
    """构造中心词的完整静态实体，但移除当前修饰语子树。"""
    ignored_words = {"多少", "几", "个", "起", "次", "张"}
    indices = sorted(static_subtree_indices(analysis, head_index) - excluded)
    blocked_words = blocked_words or set()
    value = "".join(
        analysis.words[index]
        for index in indices
        if analysis.words[index] not in ignored_words | blocked_words and analysis.pos[index] != "NT"
    )
    return normalize_entity_suffix(value)


def marked_cause(
    analysis: object, predicate_index: int, blocked_words: set[str] | None = None
) -> str | None:
    """恢复带标记或可由DEP主语证实的局部因果片段。"""
    blocked_words = blocked_words or set()
    lower_bound = 0
    for index in range(predicate_index - 1, -1, -1):
        if analysis.words[index] in {"，", ",", "。", ".", "？", "?", "；", ";"}:
            lower_bound = index + 1
            break
    marker = next(
        (index for index in range(predicate_index - 1, lower_bound - 1, -1) if analysis.words[index] in CAUSE_MARKERS),
        None,
    )
    if marker is not None:
        return "".join(
            word for word in analysis.words[marker : predicate_index + 1] if word not in blocked_words
        )

    # 无“由于”等标记时，仅采用因果谓词的直接主语/宾语子树；
    # 已删除的维度词不参与原因片段，避免“全球运营商…导致”。
    cause_heads = children(analysis, predicate_index, {"nsubj", "dobj"})
    for cause_head in cause_heads:
        if cause_head >= predicate_index:
            continue
        cause = nominal_phrase(analysis, cause_head, blocked_words)
        if cause:
            return f"{cause}{analysis.words[predicate_index]}"
    return None


def normalize_entity_suffix(value: str) -> str:
    """统计后缀不是业务实体本体，例如“管理升级单数”还原为“管理升级单”。"""
    if not value:
        return "—"
    if value in {"数", "数量", "次数", "个数"}:
        return "—"
    if len(value) > 1 and value.endswith("数量"):
        return value[:-2]
    if len(value) > 1 and value.endswith("次数"):
        return value[:-2]
    if len(value) > 1 and value.endswith("数"):
        return value[:-1]
    # “率”是指标，不是可被状态/原因筛选的业务实体。
    if value.endswith("率"):
        return "—"
    return value


def nominal_rcmod_de_candidate(
    analysis: object, predicate_index: int, dimension_words: set[str]
) -> tuple[str, str] | None:
    """补偿被误标为rcmod的“复合名词 + 的 + 名词”结构。

    仅接受名词性rcmod，且“的”为其cpm子节点、rcmod下存在名词性dobj/nn子节点。
    这能覆盖“交付风险项目的TOP3原因”，而不会吞并“产品质量导致的事故”等动词性定语。
    """
    if analysis.pos[predicate_index].startswith("V"):
        return None
    de_indices = [
        child
        for child in children(analysis, predicate_index, {"cpm"})
        if analysis.words[child] == "的" and child > predicate_index
    ]
    if not de_indices:
        return None
    de_index = min(de_indices)
    component_children = [
        child
        for child in children(analysis, predicate_index, {"dobj", "nn", "amod", "ordmod"})
        if child < de_index
    ]
    if not component_children:
        return None

    selected = {predicate_index}
    for child in component_children:
        selected.update(static_subtree_indices(analysis, child))
    modifier = "".join(
        analysis.words[index]
        for index in sorted(selected)
        if analysis.words[index] not in dimension_words
        and not is_time_token(analysis.words[index], analysis.pos[index])
    )
    head_index = analysis.heads[predicate_index] - 1
    entity = expanded_entity_phrase(analysis, head_index, dimension_words)
    if not modifier or entity == "—":
        return None
    return modifier, entity


def is_metric_head(word: str) -> bool:
    """用稳定的中文指标形态识别指标中心词，不依赖业务词典。"""
    return word in {"数", "数量", "次数", "进度"} or word.endswith(
        ("率", "度", "数量", "次数", "进度")
    )


def recovered_locative_metric_modifiers(
    analysis: object, metric_head_index: int, dimension_words: set[str]
) -> list[str]:
    """恢复被分词器拆开的“网上/线下/项目内”等局部片段。

    仅接受 ``名词 -[lobj]-> 方位词 -[dep]-> 指标中心词`` 的连续结构，
    并要求方位词紧跟其名词。这样不会把普通的“项目中”范围随意并入指标，
    但能补偿“网 / 上 / 事故 / 数”这一类错误切分。
    """
    recovered: list[str] = []
    for suffix_index, word in enumerate(analysis.words):
        if word not in LOCATIVE_COMPOUND_SUFFIXES:
            continue
        if analysis.heads[suffix_index] != metric_head_index + 1 or analysis.labels[suffix_index] != "dep":
            continue
        left_parts = children(analysis, suffix_index, {"lobj"})
        if len(left_parts) != 1:
            continue
        left_index = left_parts[0]
        if left_index != suffix_index - 1 or not analysis.pos[left_index].startswith("N"):
            continue
        if analysis.words[left_index] in dimension_words:
            continue
        recovered.append(f"{analysis.words[left_index]}{word}")
    return recovered


def metric_name_parts(analysis: object, head_index: int, dimension_words: set[str]) -> list[str]:
    """按词序展开指标名的nn/amod/ordmod内部链，不纳入“X的”范围。"""
    selected = {head_index}
    frontier = [head_index]
    while frontier:
        parent = frontier.pop()
        for child in children(analysis, parent):
            is_level_classifier = (
                analysis.labels[child] == "clf"
                and analysis.words[child] == "级"
                and bool(children(analysis, child, {"ordmod"}))
            )
            if analysis.labels[child] not in {"nn", "amod", "ordmod"} and not is_level_classifier:
                continue
            selected.add(child)
            frontier.append(child)
    indices = [
        index
        for index in sorted(selected)
        if analysis.words[index] not in dimension_words
        and not is_time_token(analysis.words[index], analysis.pos[index])
    ]
    parts: list[str] = []
    position = 0
    while position < len(indices):
        level_end = position
        while level_end < len(indices) and analysis.pos[indices[level_end]] == "OD":
            level_end += 1
        if (
            level_end > position
            and level_end < len(indices)
            and analysis.words[indices[level_end]] == "级"
        ):
            parts.append("".join(analysis.words[index] for index in indices[position : level_end + 1]))
            position = level_end + 1
            continue
        parts.append(analysis.words[indices[position]])
        position += 1
    return parts


def metric_entity_subtree_indices(analysis: object, head_index: int) -> set[int]:
    """统计指标前的业务实体子树；不纳入“X的”范围 assmod。"""
    selected = {head_index}
    frontier = [head_index]
    while frontier:
        parent = frontier.pop()
        for child in children(analysis, parent):
            is_level_classifier = (
                analysis.labels[child] == "clf"
                and analysis.words[child] == "级"
                and bool(children(analysis, child, {"ordmod"}))
            )
            if analysis.labels[child] not in {"nn", "amod", "ordmod", "conj", "cc"} and not is_level_classifier:
                continue
            selected.add(child)
            frontier.append(child)
    return selected


def statistical_metric_entity_candidates(
    analysis: object, dimension_words: set[str]
) -> list[tuple[str, str]]:
    """从“高风险客户数”恢复“高风险 → 客户”，而非“客户 → 数”。"""
    candidates: list[tuple[str, str]] = []
    statistic_heads = {"数", "数量", "次数"}
    for metric_index, metric_word in enumerate(analysis.words):
        if metric_word not in statistic_heads or metric_word in dimension_words:
            continue
        selected = metric_entity_subtree_indices(analysis, metric_index) - {metric_index}
        if not selected or any(analysis.words[index] in dimension_words for index in selected):
            continue

        condition_groups: list[set[int]] = []
        # “一二级事故数”中的“一二级”作为完整等级条件，而不是拆成“级”。
        for index in selected:
            if (
                analysis.labels[index] == "clf"
                and analysis.words[index] == "级"
                and children(analysis, index, {"ordmod"})
            ):
                condition_groups.append({index, *children(analysis, index, {"ordmod"})})
        # “高风险客户数”“高危变更操作次数”中的形容词性条件。
        for index in selected:
            if analysis.labels[index] in {"amod", "ordmod"} and not any(
                index in group for group in condition_groups
            ):
                condition_groups.append({index})

        for group in condition_groups:
            modifier = "".join(analysis.words[index] for index in sorted(group))
            entity = "".join(
                analysis.words[index]
                for index in sorted(selected - group)
                if not is_time_token(analysis.words[index], analysis.pos[index])
            )
            entity = normalize_entity_suffix(entity)
            if modifier and entity != "—":
                candidates.append((modifier, entity))
    return candidates


def metric_surface(analysis: object, dimension_words: set[str]) -> str:
    """获得去时间、去维度后、止于问句谓词前的连续指标文本。"""
    stop_words = {
        "是", "有", "多少", "几", "最", "最多", "最低", "最高",
        "小于", "低于", "大于", "高于", "同比", "环比", "增加", "减少", "上升", "下降",
        "需", "需要", "应", "要", "待",
    }
    values: list[str] = []
    # “哪几个地区部的…”“哪个产品线的…”的前半段是待枚举维度。
    # 不论维度差异文件是否恰好删掉它，都应从第一个“的”之后再取指标短语。
    first_content = next(
        (
            word
            for word, pos in zip(analysis.words, analysis.pos, strict=True)
            if word not in dimension_words
            and not is_time_token(word, pos)
            and word not in {"，", ",", "。", ".", "？", "?"}
        ),
        "",
    )
    skip_enum_prefix = first_content in {"哪", "哪个", "哪些", "哪几个"}
    for index, (word, pos) in enumerate(zip(analysis.words, analysis.pos, strict=True)):
        # “业务中，哪个产品线的…”前面可能残留方位词“中”。遇到新的
        # 枚举词时重新开始，避免把它当成指标对象的一部分。
        if word in {"哪", "哪个", "哪些", "哪几个"}:
            values.clear()
            skip_enum_prefix = True
            continue
        if skip_enum_prefix:
            if word == "的":
                skip_enum_prefix = False
            continue
        if word in {"，", ",", "？", "?", "。", "."}:
            # 前半句是时间/范围，逗号后才出现“哪个/哪些”时，继续扫描。
            if any(
                future in {"哪", "哪个", "哪些", "哪几个"}
                for future in analysis.words[index + 1 :]
            ):
                values.clear()
                continue
            if values:
                break
            continue
        # “有哪些/有多少”中的句首“有”不是指标结束；只有已经开始收集
        # 指标文本后才可作为“有多少”的问句谓词。
        if word in stop_words and not (word == "有" and not values):
            break
        if word in {"截止", "截至"} or word in dimension_words or is_time_token(word, pos):
            continue
        values.append(word)
    return "".join(values)


def clean_metric_object(value: str) -> str:
    """移除问句残片及“业务的”一类范围前缀，保留指标左侧对象。"""
    # “业务的服务订货…”、“哪个产品线的网上事故…”只取最后一个“的”之后
    # 的局部对象；“小国小网的网络”则去掉结构词“的”并保留两侧实体成分。
    if "的" in value:
        before, after = value.rsplit("的", 1)
        is_scope_prefix = (
            before.startswith(GENERIC_OBJECT_PREFIXES)
            or bool(re.match(r"^(?:有哪些|哪个|哪些|哪几个|哪)", before))
        )
        is_condition_prefix = any(trigger in before for trigger in DE_CLAUSE_TRIGGERS)
        # 条件短语已有“修饰 → 实体”列承载：
        # “健康的小国小网网络总数”应切为“小国小网网络 → 总数”。
        # 若“的”恰在指标前（如“国家三领先的达成率”），after为空，
        # 左侧仍是被度量对象，不能删除。
        value = after if after and (is_scope_prefix or is_condition_prefix) else before + after
    value = re.sub(r"^(?:有哪些|哪个|哪些|哪几个|哪|请提供|提供)+", "", value)
    for prefix in GENERIC_OBJECT_PREFIXES:
        if value.startswith(prefix) and len(value) > len(prefix):
            value = value[len(prefix) :]
            break
    return value.strip("，, ")


def contextual_quality_metric(
    analysis: object, dimension_words: set[str], surface: str
) -> tuple[str, str] | None:
    """仅在“对象质量 + 评价谓词”语境下恢复“对象 → 质量”。

    “产品质量导致事故”中的“质量”是原因短语的一部分，不能视作指标；
    “工程实施质量需提升”则以“质量”为评价中心，可安全恢复。
    """
    if not surface.endswith("质量") or len(surface) <= len("质量"):
        return None
    quality_index = max(
        (index for index, word in enumerate(analysis.words) if word == "质量"),
        default=-1,
    )
    if quality_index < 0:
        return None
    evaluation_words = {"需", "需要", "应", "待", "提升", "改进", "达标", "低于", "高于"}
    if not any(word in evaluation_words for word in analysis.words[quality_index + 1 :]):
        return None
    entity = clean_metric_object(surface[: -len("质量")])
    return (entity, "质量") if entity else None


def object_metric_relations(analysis: object, dimension_words: set[str]) -> str:
    """恢复“被度量业务对象 → 指标”，每个指标最多产生一条关系。

    以去维度后的连续表面文本为主、最长指标尾部为锚点，因此即使DEP将
    “超期未关闭率”切成rcmod，也可恢复“假设 → 超期未关闭率”。
    """
    relations: list[str] = []
    seen: set[str] = set()

    def add(entity: str, indicator: str) -> None:
        relation = f"{entity} → {indicator}"
        if entity and indicator and entity != indicator and relation not in seen:
            seen.add(relation)
            relations.append(relation)

    value = metric_surface(analysis, dimension_words)
    for indicator in INDICATOR_TAILS:
        if not value.endswith(indicator) or len(value) <= len(indicator):
            continue
        entity = clean_metric_object(value[: -len(indicator)])
        add(entity, indicator)
        break
    if not relations:
        quality_metric = contextual_quality_metric(analysis, dimension_words, value)
        if quality_metric is not None:
            add(*quality_metric)
    return "；".join(relations) or "—"


def query_intent(analysis: object, dimension_words: set[str]) -> str:
    """识别问法类型；该信息单列保存，不参与实体或条件抽取。"""
    text = "".join(
        word
        for word, pos in zip(analysis.words, analysis.pos, strict=True)
        if word not in dimension_words and not is_time_token(word, pos)
    )
    value_indicator_tails = (
        "风险超期未关闭率", "预算完成率", "预测完成率", "恢复及时率",
        "超期未关闭率", "管理成熟度", "达成率", "成本率", "完成率",
        "成功率", "及时率", "成熟度", "占比",
    )
    if any(text.endswith(tail) or f"{tail}是" in text for tail in value_indicator_tails):
        return "指标值查询"
    # “有多少个项目存在风险”主问题是计数，而不是存在性判断；只有无数量
    # 量词时，才把“存在”解释为布尔查询。
    has_count_question = bool(
        re.search(r"(?:有|共|总共|发生了?|还有)?多少(?:个|张|起|次|项|例)?", text)
    )
    if has_count_question:
        return "数量查询"
    if any(marker in text for marker in ("是否", "有没有", "有无", "存在")):
        return "存在性查询"
    if any(marker in text for marker in ("哪个", "哪些", "哪几个")):
        return "枚举/维度查询"
    if any(marker in text for marker in ("最多", "最少", "最高", "最低")):
        return "极值查询"
    if any(marker in text for marker in ("多少", "几", "数量", "次数", "总数", "数")):
        return "数量查询"
    if any(marker in text for marker in ("详情", "明细", "详细信息")):
        return "详情查询"
    return "其他查询"


def has_implicit_count_extremum(analysis: object) -> bool:
    """“哪个…最多/最少”隐含按对象数量排序，而非指标值极值。"""
    text = "".join(analysis.words)
    return "最多" in text or "最少" in text or ("最" in text and any(word in {"多", "少"} for word in analysis.words))


def relation_pairs(value: str) -> list[tuple[str, str]]:
    if value == "—":
        return []
    pairs: list[tuple[str, str]] = []
    for relation in value.split("；"):
        if " → " not in relation:
            continue
        left, right = relation.split(" → ", 1)
        if left and right:
            pairs.append((left, right))
    return pairs


def normalize_object_metric_with_conditions(
    object_metric_value: str, condition_relations: str
) -> str:
    """用已识别条件的实体边界校正“对象 → 指标”的左侧对象。"""
    if object_metric_value == "—":
        return "—"
    condition_pairs = relation_pairs(condition_relations)
    normalized: list[str] = []
    seen: set[str] = set()
    for object_value, indicator in relation_pairs(object_metric_value):
        entity = object_value
        for modifier, target in condition_pairs:
            if target.endswith(tuple(INDICATOR_TAILS)):
                continue
            # “未关闭二级管理升级 → 数量”：对象以主实体结尾时，移除条件。
            if object_value != target and object_value.endswith(target):
                entity = target
                break
            # “网络质量保障的项目数”：对象等于条件短语时，以右侧项目为主实体。
            if object_value == modifier:
                entity = target
                break
            # “交付高风险项目数”中，“项目”被吸收到指标“项目数”，
            # 而“高风险”被吸收到对象。若条件右侧正是指标前缀名词，
            # 便可由两侧证据恢复“交付项目 → 项目数”。
            if (
                is_semantic_condition(modifier)
                and target
                and indicator.startswith(target)
                and modifier in object_value
            ):
                entity = normalize_entity_suffix(object_value.replace(modifier, "") + target)
                break
        relation = f"{entity} → {indicator}"
        if relation not in seen:
            seen.add(relation)
            normalized.append(relation)
    return "；".join(normalized) or "—"


def normalize_condition_entities_with_metric(
    condition_relations: str, object_metric_value: str
) -> str:
    """以对象—指标关系补齐被DEP拆开的条件中心实体。

    例如“解决方案相关的交付高风险项目数”可分别得到
    “解决方案相关 → 交付”和“高风险 → 项目”，但对象—指标已恢复为
    “交付项目 → 项目数”。此时两个条件的中心词都应回指“交付项目”。
    """
    metric_entities = [entity for entity, _indicator in relation_pairs(object_metric_value)]
    if not metric_entities:
        return condition_relations
    values: list[str] = []
    seen: set[str] = set()
    for modifier, entity in relation_pairs(condition_relations):
        replacement = next(
            (
                metric_entity
                for metric_entity in metric_entities
                if entity != metric_entity and entity in metric_entity
            ),
            entity,
        )
        relation = f"{modifier} → {replacement}"
        if relation not in seen:
            seen.add(relation)
            values.append(relation)
    return "；".join(values) or "—"


def count_question_entity_candidate(analysis: object, dimension_words: set[str]) -> str:
    """从“多少个/次/张 + 实体”局部结构恢复数量问句的主实体。"""
    classifiers = {"个", "次", "张", "起", "项", "例"}
    stop_words = {"是", "有", "存在", "分别", "什么", "等级", "？", "?", "，", ",", "。", "."}
    for position, word in enumerate(analysis.words):
        if word not in {"多少", "几"}:
            continue
        index = position + 1
        while index < len(analysis.words) and analysis.words[index] in classifiers:
            index += 1
        selected: list[str] = []
        while index < len(analysis.words):
            token = analysis.words[index]
            if token in stop_words:
                break
            if token in dimension_words or is_time_token(token, analysis.pos[index]):
                index += 1
                continue
            compound = local_locative_compound_at(analysis, index)
            if compound is not None:
                end, value = compound
                if not any(analysis.words[position] in dimension_words for position in range(index, end + 1)):
                    selected.append(value)
                index = end + 1
                continue
            if (
                analysis.pos[index].startswith("N")
                or token in NOMINAL_COMPOUND_PARTS
                or token in RISK_WORDS
                or analysis.labels[index] in {"dobj", "nsubj", "top"}
                or re.fullmatch(r"(?:[一二三四五六七八九十]+|[A-Z])级(?:以上|以下)?", token)
            ):
                selected.append(token)
                index += 1
                continue
            # 高风险/未关闭等条件可能被POS标作形容词或动词，跳过但继续寻找实体。
            if token in {"未", "不", "高", "中", "低", "重大", "风险", "关闭"}:
                index += 1
                continue
            break
        value = normalize_entity_suffix("".join(selected))
        if value != "—":
            # 由条件关系再剥离“高风险、二级、未关闭”等前缀。
            return value
        # “订货完成了多少”中，实体在“多少”左侧，作为谓词的主语出现。
        if analysis.heads[position] != 0:
            predicate = analysis.heads[position] - 1
            subjects = children(analysis, predicate, {"nsubj", "top", "dobj"})
            if subjects:
                value = expanded_entity_phrase(analysis, subjects[0], dimension_words)
                if value != "—":
                    return value
    return "—"


def detail_query_entity_candidate(analysis: object, dimension_words: set[str]) -> str:
    """从“提供/查看 X 的详细信息”中恢复 X，而非把“详细信息”当实体。"""
    detail_words = {"详情", "明细", "信息"}
    ignored = {"这", "该", "这些", "上述", "个", "张", "起", "次", "项", "例", "详细"}
    for target, word in enumerate(analysis.words):
        if word not in detail_words:
            continue
        index = target - 1
        # “风险项目的详细信息”：跳过“详细”和结构词“的”。
        while index >= 0 and analysis.words[index] in ignored | {"的"}:
            index -= 1
        selected: list[str] = []
        while index >= 0:
            token = analysis.words[index]
            if token in {"请", "提供", "查看", "展示", "列出", "，", ",", "。", ".", "？", "?"}:
                break
            if token in dimension_words or is_time_token(token, analysis.pos[index]) or token in ignored:
                index -= 1
                continue
            if (
                analysis.pos[index].startswith("N")
                or token in NOMINAL_COMPOUND_PARTS
                or token in RISK_WORDS
                or re.fullmatch(r"(?:[一二三四五六七八九十]+|[A-Z])级(?:以上|以下)?", token)
            ):
                selected.append(token)
                index -= 1
                continue
            break
        value = normalize_entity_suffix("".join(reversed(selected)))
        if value != "—":
            return value
    return "—"


def business_entity_candidates(
    analysis: object,
    dimension_words: set[str],
    condition_relations: str,
    object_metric_relations_value: str,
) -> str:
    """实体优先候选：优先采用已验证关系的右侧实体/对象，再用名词短语兜底。"""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = normalize_entity_suffix(value)
        if value in GENERIC_OBJECT_PREFIXES:
            return
        for prefix in GENERIC_OBJECT_PREFIXES:
            if value.startswith(prefix) and len(value) > len(prefix):
                value = value[len(prefix) :]
                break
        if value != "—" and value not in seen:
            seen.add(value)
            candidates.append(value)

    condition_pairs = relation_pairs(condition_relations)
    condition_modifiers = {modifier for modifier, _entity in condition_pairs}
    indicator_condition_modifiers = {
        modifier
        for modifier, entity in condition_pairs
        if entity.endswith(tuple(INDICATOR_TAILS))
    }
    metric_entities = {
        entity for entity, _indicator in relation_pairs(object_metric_relations_value)
    }
    # 条件关系右侧通常是中心业务实体，优先级高于对象—指标左侧；
    # 但“EHS → 管理成熟度”中的右侧属于指标，不应误列为业务实体。
    for _modifier, entity in condition_pairs:
        if not entity.endswith(tuple(INDICATOR_TAILS)):
            # 若对象—指标已经恢复出更完整的实体（“交付项目”），
            # 不再并列输出其被DEP切碎的组成片段（“交付”“项目”）。
            if any(entity != metric_entity and entity in metric_entity for metric_entity in metric_entities):
                continue
            add(entity)
    for entity, _indicator in relation_pairs(object_metric_relations_value):
        # 若对象已经作为“条件 → 实体”中的条件，主实体应采用右侧实体，
        # 如“网络质量保障 → 项目”，而非把“网络质量保障”排在项目之前。
        if entity not in condition_modifiers or entity in indicator_condition_modifiers:
            add(entity)

    # 没有语义关系时，才退回到局部名词短语，避免输出一长串重叠候选。
    if not candidates:
        for index, word in enumerate(analysis.words):
            if (
                not analysis.pos[index].startswith("N")
                or word in dimension_words
                or is_time_token(word, analysis.pos[index])
                or is_metric_head(word)
            ):
                continue
            if analysis.heads[index] != 0:
                parent = analysis.heads[index] - 1
                if analysis.pos[parent].startswith("N") and analysis.labels[index] in {"nn", "amod", "assmod"}:
                    continue
            add(expanded_entity_phrase(analysis, index, dimension_words))
    return "；".join(candidates) or "—"


def indicator_candidates(object_metric_relations_value: str) -> str:
    """从对象—指标关系中输出去重后的指标候选。"""
    values: list[str] = []
    seen: set[str] = set()
    for _entity, indicator in relation_pairs(object_metric_relations_value):
        if indicator not in seen:
            seen.add(indicator)
            values.append(indicator)
    return "；".join(values) or "—"


def bare_rcmod_prefix(
    analysis: object, predicate_index: int, head_index: int, blocked_words: set[str]
) -> str | None:
    """识别被误标为rcmod、但实际嵌在复合名词中的左侧修饰片段。

    例如“高风险交付售前项目”中，模型可能给出
    高 -amod→ 风险 -dep→ 交付 -rcmod→ 项目。
    “交付”没有“的”、状语、否定或主谓宾论元，且其左侧仍有名词性片段，
    因而可将“高风险”安全地视作“交付售前项目”的修饰语。
    """
    # 有“的”或句法论元时是真正的动词性定语，不把它当复合名词的一部分。
    if "的" in analysis.words[predicate_index + 1 : head_index]:
        return None
    clause_labels = {"dobj", "advmod", "neg", "prep", "pobj"}
    # 某些业务名词（如“交付”）会被DEP错误地赋予 rcmod，
    # 其左侧“风险”又被错误标为 nsubj。只有中心词本身是动词时，
    # 才把 nsubj 当作动词论元而拒绝该补偿。
    if analysis.pos[predicate_index].startswith("V"):
        clause_labels.add("nsubj")
    if children(analysis, predicate_index, clause_labels):
        return None

    allowed = {"dep", "nn", "amod", "assmod", "ordmod", "conj", "cc"}
    if not analysis.pos[predicate_index].startswith("V"):
        allowed.add("nsubj")
    selected: set[int] = set()
    frontier = [
        child
        for child in children(analysis, predicate_index, allowed)
        if child < predicate_index
    ]
    while frontier:
        child = frontier.pop()
        if child in selected:
            continue
        selected.add(child)
        frontier.extend(children(analysis, child, allowed))

    # 必须有谓词左侧、且与之相连的名词性片段；否则不做补偿。
    if not selected:
        return None
    value = "".join(
        analysis.words[index]
        for index in sorted(selected)
        if analysis.words[index] not in blocked_words and analysis.pos[index] != "NT"
    )
    return value or None


def noun_chain_components(
    analysis: object, head_index: int, dimension_words: set[str]
) -> list[str]:
    """将名词链拆为可读组件；nn递归展开，amod与其中心词保持为一个条件组件。"""
    components: list[str] = []
    for child in sorted(children(analysis, head_index, {"nn"})):
        components.extend(noun_chain_components(analysis, child, dimension_words))
    adjective = "".join(
        analysis.words[child]
        for child in sorted(children(analysis, head_index, {"amod", "ordmod"}))
        if analysis.words[child] not in dimension_words
    )
    if analysis.words[head_index] not in dimension_words and not is_time_token(
        analysis.words[head_index], analysis.pos[head_index]
    ):
        components.append(f"{adjective}{analysis.words[head_index]}")
    return components


def coordinated_static_component(
    analysis: object, root_index: int, dimension_words: set[str]
) -> str:
    """恢复一个静态成分；补上DEP误挂在并列首项下的另一半条件。

    例如“重大或高风险”中，HanLP 有时给出
    ``高风险 -[dep]-> 重大``，而非 ``conj``。出现连词时仅接纳该类
    ``dep/conj``，避免把一般 dep 关系误并入名词短语。
    """
    selected = static_subtree_indices(analysis, root_index)
    if children(analysis, root_index, {"cc"}):
        for child in children(analysis, root_index, {"dep", "conj"}):
            selected.update(static_subtree_indices(analysis, child))
            selected.add(child)
    return "".join(
        analysis.words[index]
        for index in sorted(selected)
        if analysis.words[index] not in dimension_words
    )


def merge_coordinated_components(
    analysis: object, components: list[tuple[int, str]]
) -> list[str]:
    """把相邻的“高风险 / 或 / 中风险”收成一个完整条件。"""
    merged: list[tuple[int, str]] = []
    for index, value in sorted(components):
        value = value.strip("或和及")
        if not value:
            continue
        if merged:
            previous_index, previous = merged[-1]
            between = analysis.words[previous_index + 1 : index]
            markers = "".join(word for word in between if word in COORDINATION_MARKERS)
            # 连词必须是两组件之间唯一的实词；否则“重大或高风险 + 售前”
            # 会把相距较远的“售前”误并到并列条件中。
            if markers and all(word in COORDINATION_MARKERS for word in between):
                merged[-1] = (previous_index, f"{previous}{markers}{value}")
                continue
        merged.append((index, value))
    return [value for _index, value in merged]


def risk_condition_around(analysis: object, risk_index: int) -> str:
    """不依赖并列DEP标签，按连续表面片段恢复“高风险或中风险”。"""
    selected = {risk_index}
    for step in (-1, 1):
        index = risk_index + step
        while 0 <= index < len(analysis.words):
            word = analysis.words[index]
            if word in RISK_WORDS or word in COORDINATION_MARKERS:
                selected.add(index)
                index += step
                continue
            break
    # “重大或高风险”里“重大”不是风险词，但它与风险等级由连词直接连接，
    # 因此同属一个选择条件。只接纳连词左侧紧邻的一个形容/名词，范围受限。
    leftmost = min(selected)
    # 左向扫描会先收进“或”，此时再把其左侧的“重大”等并列项补入。
    if leftmost > 0 and analysis.words[leftmost] in COORDINATION_MARKERS:
        preceding = leftmost - 1
        # CTB POS 会把“重大”标成 VA（形容词性谓词）。
        if analysis.pos[preceding].startswith(("A", "J", "N", "V")):
            selected.add(preceding)
    return "".join(analysis.words[index] for index in sorted(selected))


def attached_att_predicate(analysis: object, index: int) -> int | None:
    """从并列/误挂词向上找到其rcmod/vmod锚点，最多追两跳。"""
    current = index
    for _ in range(2):
        if analysis.labels[current] in {"rcmod", "vmod"} and analysis.heads[current] != 0:
            return current
        if analysis.heads[current] == 0:
            return None
        current = analysis.heads[current] - 1
    return None


def postposed_risk_state(analysis: object, predicate_index: int, dimension_words: set[str]) -> tuple[str, str] | None:
    """恢复句末“NPX保障项目存在高风险或中风险”的状态条件。

    仅处理谓词后直到分句末尾都是风险条件、且其中没有“的”的情况；带“的”
    的前置定语交由 DEP rcmod 路径处理，避免跨结构猜测。
    """
    if analysis.words[predicate_index] != "存在":
        return None
    stop = next(
        (
            index
            for index in range(predicate_index + 1, len(analysis.words))
            if analysis.words[index] in {"，", ",", "。", ".", "？", "?", "；", ";"}
        ),
        len(analysis.words),
    )
    tail = analysis.words[predicate_index + 1 : stop]
    if "的" in tail:
        return None
    risk_indices = [
        index for index in range(predicate_index + 1, stop) if analysis.words[index] in RISK_WORDS
    ]
    if not risk_indices:
        return None
    # 以谓词左侧最近的名词为实体中心词，并由DEP展开局部名词短语。
    head_index = next(
        (
            index
            for index in range(predicate_index - 1, -1, -1)
            if analysis.pos[index].startswith("N") and analysis.words[index] not in dimension_words
        ),
        None,
    )
    if head_index is None:
        return None
    entity = expanded_entity_phrase(analysis, head_index, dimension_words)
    if entity == "—":
        return None
    return f"存在{risk_condition_around(analysis, risk_indices[0])}", entity


def entity_after_risk_condition(
    analysis: object, risk_index: int, head_index: int, dimension_words: set[str]
) -> str:
    """在无“的”的紧邻复合名词中保留风险条件右侧的实体成分。

    例如 DEP 将“重大”挂为 rcmod、将“交付”挂为 vmod 时，仍能从
    “重大或高风险交付EI项目”恢复 ``交付EI项目``。含“的”的结构不走
    此路径，避免把真正的定语从句误拼入实体。
    """
    if risk_index >= head_index:
        return expanded_entity_phrase(analysis, head_index, dimension_words)
    indices = list(range(risk_index + 1, head_index + 1))
    if any(analysis.words[index] == "的" for index in indices):
        return expanded_entity_phrase(analysis, head_index, dimension_words)
    if not indices or not all(analysis.pos[index].startswith(("N", "V", "A", "J")) for index in indices):
        return expanded_entity_phrase(analysis, head_index, dimension_words)
    value = "".join(
        analysis.words[index]
        for index in indices
        if analysis.words[index] not in dimension_words
        and not is_time_token(analysis.words[index], analysis.pos[index])
    )
    return normalize_entity_suffix(value) if value else expanded_entity_phrase(analysis, head_index, dimension_words)


def nominalized_compound_components(
    analysis: object, predicate_index: int, dimension_words: set[str]
) -> list[str] | None:
    """恢复“比拼网络”“交付高风险项目”等被误作动宾的复合名词链。"""
    if analysis.words[predicate_index] not in NOMINALIZED_COMPOUND_PREDICATES:
        return None
    items: list[tuple[int, list[str]]] = []
    used_indices: set[int] = set()

    def add_item(index: int, parts: list[str]) -> None:
        if parts and index not in used_indices:
            used_indices.add(index)
            items.append((index, parts))

    for child in children(analysis, predicate_index, {"nsubj", "dobj"}):
        parts = noun_chain_components(analysis, child, dimension_words)
        add_item(child, parts)
    # “高风险交付项目”常把“高风险”误标为交付的 advmod；这里仅接纳
    # 三个风险等级，不能泛化为把所有状语都塞进实体。
    for child in children(analysis, predicate_index, {"advmod"}):
        if analysis.words[child] in RISK_WORDS:
            add_item(child, [analysis.words[child]])
    label = analysis.labels[predicate_index]
    if label in {"rcmod", "vmod"} and analysis.heads[predicate_index] != 0:
        head_index = analysis.heads[predicate_index] - 1
        if analysis.pos[head_index].startswith("N"):
            # 不能把完整 head 子树和其子节点同时加入，否则会产生
            # “售前售前项目”“EIEI项目”。这里按实际词位拆开。
            # 谓词右侧的静态成分属于同一个业务实体：
            # “交付 + 高风险 + 项目”“交付 + 售前 + 项目”。
            for child in children(analysis, head_index, {"nn", "amod", "assmod"}):
                if child > predicate_index:
                    add_item(child, noun_chain_components(analysis, child, dimension_words))
            add_item(head_index, [analysis.words[head_index]])
    if not items:
        return None
    items.append((predicate_index, [analysis.words[predicate_index]]))
    components = [part for _index, parts in sorted(items) for part in parts]
    # 至少三个组件且谓词不在末尾；允许“交付高风险项目”这类谓词位于短语开头，
    # 但不把普通的两词“交付项目”动词短语误合并。
    predicate_position = components.index(analysis.words[predicate_index])
    if len(components) < 3 or predicate_position == len(components) - 1:
        return None
    return components


def is_expandable_compound_child(analysis: object, child_index: int, head_index: int) -> bool:
    """判断rcmod/vmod是否其实是复合业务实体成分，而非真实从句。"""
    if analysis.words[child_index] not in NOMINALIZED_COMPOUND_PREDICATES:
        return False
    if analysis.labels[child_index] not in {"rcmod", "vmod"}:
        return False
    if "的" in analysis.words[child_index + 1 : head_index]:
        return False
    # 否定、介词、宾语型状语通常说明它是真实谓词；风险等级 advmod
    # 与 EI 等名词性dobj是本项目中稳定的复合实体误标模式，允许保留。
    if children(analysis, child_index, {"neg", "prep", "pobj"}):
        return False
    for child in children(analysis, child_index, {"advmod"}):
        if analysis.words[child] not in RISK_WORDS:
            return False
    return True


def is_entity_condition_component(word: str, label: str) -> bool:
    """判断词是否是业务实体之外的等级/风险条件。"""
    if word in ENTITY_CONDITION_WORDS:
        return True
    # A级、S级、一二级等是等级条件；“升级”不会匹配该模式。
    if re.fullmatch(r"(?:[A-Z]|[一二三四五六七八九十]+)级", word):
        return True
    return label == "ordmod"


def is_semantic_condition(modifier: str) -> bool:
    """判定可作为RAG筛选条件的修饰语，用于删除其冗余原子链。"""
    return (
        modifier in RISK_WORDS | {"高危", "低危", "重大"}
        # 仅完整的等级短语才是条件；“二级管理”不能整体当作条件，
        # 应由后续规则恢复为“二级 → 管理升级”。
        or bool(re.fullmatch(r"(?:[A-Z]|[一二三四五六七八九十]+)级(?:以上|以下)?", modifier))
        or modifier.startswith(("存在", "未", "不", "有风险"))
        or modifier.endswith(("导致", "造成", "引起", "产生"))
    )


def is_business_identifier(value: str) -> bool:
    """识别无需业务词典即可高置信保留的产品/项目标识。

    P3、5G、NPX 等字母数字混合或全大写缩写，以及 FaceboY、SmartCare
    这类含多个大写边界的驼峰专名，通常是业务限定词；普通“交付/项目”
    等中文结构词不会命中。
    """
    if not re.fullmatch(r"[A-Za-z0-9]+", value) or not any(char.isalpha() for char in value):
        return False
    has_digit = any(char.isdigit() for char in value)
    uppercase_count = sum(char.isupper() for char in value)
    return has_digit or (value.isupper() and len(value) >= 2) or uppercase_count >= 2


def consume_level_unit(words: list[str], start: int) -> tuple[int, str] | None:
    """读取一个未必被正确合并分词的等级单元，如“二/级”或“一级”。"""
    if start >= len(words):
        return None
    word = words[start]
    if re.fullmatch(r"(?:[一二三四五六七八九十两]+|[A-Za-z]|\d+)级", word):
        return start, word

    end = start
    number_parts: list[str] = []
    while end < len(words) and (
        words[end] in LEVEL_NUMBER_TOKENS or re.fullmatch(r"\d+", words[end])
    ):
        number_parts.append(words[end])
        end += 1
    if number_parts and end < len(words) and words[end] == "级":
        return end, "".join(number_parts) + "级"
    return None


def level_expression_at(words: list[str], start: int) -> tuple[int, str] | None:
    """恢复“一级和二级”“二级以上”等连续等级条件。"""
    first = consume_level_unit(words, start)
    if first is None:
        return None
    end, value = first
    if end + 1 < len(words) and words[end + 1] in LEVEL_RANGE_SUFFIXES:
        end += 1
        value += words[end]

    # 只有等级单元紧邻连接词时才合并；“一级管理升级和二级管理升级”
    # 会自然拆成两条“一级/二级 → 管理升级”。
    while end + 1 < len(words) and words[end + 1] in LEVEL_CONNECTORS:
        next_unit = consume_level_unit(words, end + 2)
        if next_unit is None:
            break
        next_end, next_value = next_unit
        value += words[end + 1] + next_value
        end = next_end
        if end + 1 < len(words) and words[end + 1] in LEVEL_RANGE_SUFFIXES:
            end += 1
            value += words[end]
    return end, value


def level_condition_entity(
    analysis: object, expression_end: int, dimension_words: set[str]
) -> str:
    """取等级短语右侧的局部名词实体，不依赖其DEP是否挂接正确。"""
    index = expression_end + 1
    while index < len(analysis.words) and analysis.words[index] == "的":
        index += 1

    selected: list[str] = []
    while index < len(analysis.words):
        word = analysis.words[index]
        if word in {"，", ",", "。", ".", "？", "?", "是", "有", "共", "共有", "发生", "多少", "几"}:
            break
        # 例如“一级管理升级和二级管理升级”：第一个实体在连接词前结束。
        if word in LEVEL_CONNECTORS and level_expression_at(analysis.words, index + 1):
            break
        if word == "的":
            break
        if word in dimension_words or is_time_token(word, analysis.pos[index]):
            index += 1
            continue
        compound = local_locative_compound_at(analysis, index)
        if compound is not None:
            end, value = compound
            if not any(analysis.words[position] in dimension_words for position in range(index, end + 1)):
                selected.append(value)
            index = end + 1
            continue
        if analysis.pos[index].startswith("N") or word in NOMINAL_COMPOUND_PARTS:
            selected.append(word)
            index += 1
            continue
        break
    return normalize_entity_suffix("".join(selected))


def level_condition_relations(analysis: object, dimension_words: set[str]) -> list[tuple[str, str]]:
    """从分词序列恢复等级条件到其业务实体的关系。

    DEP常把“一/级”拆开或漏连，故此处仅以连续等级形式和右侧局部名词
    边界为证据；不扫描跨介词、谓词或“的”的远距离名词，避免过度恢复。
    """
    relations: list[tuple[str, str]] = []
    index = 0
    while index < len(analysis.words):
        expression = level_expression_at(analysis.words, index)
        if expression is None:
            index += 1
            continue
        end, modifier = expression
        entity = level_condition_entity(analysis, end, dimension_words)
        if entity != "—":
            # 平面“修饰 → 实体”结果中，并列等级各自是可检索的筛选条件。
            # “一级和二级网上事故”输出两条；没有连接词的“一二级”、
            # 带范围的“二级以上”仍保留为不可拆的单一条件。
            levels = [part for part in re.split(r"[和或及、]", modifier) if part]
            for level in levels:
                relations.append((level, entity))
        index = end + 1
    return relations


def negative_rcmod_state(analysis: object, predicate_index: int) -> str | None:
    """恢复“恢复不及时”“未关闭”等否定状态型动词性定语。"""
    if analysis.labels[predicate_index] not in {"rcmod", "vmod"}:
        return None
    if not analysis.pos[predicate_index].startswith(("V", "A")):
        return None
    negations = sorted(children(analysis, predicate_index, {"neg"}))
    if not negations:
        return None
    # “恢复 -[mmod]→ 及时”这一类补足语是状态短语的一部分；只取定语
    # 谓词左侧的直接动词性修饰，避免吸收远距离业务范围。
    state_prefix = "".join(
        analysis.words[index]
        for index in sorted(children(analysis, predicate_index, {"mmod"}))
        if index < negations[0] and analysis.pos[index].startswith("V")
    )
    return state_prefix + "".join(analysis.words[index] for index in negations) + analysis.words[predicate_index]


def de_clause_entity(analysis: object, de_index: int, dimension_words: set[str]) -> str:
    """取得“X 的 Y”中Y的紧邻局部实体，不把统计后缀并入实体。"""
    metric_suffixes = {"数", "数量", "次数", "总数", "率", "占比"}
    selected: list[str] = []
    index = de_index + 1
    while index < len(analysis.words):
        word = analysis.words[index]
        if word in metric_suffixes or word in {"是", "有", "多少", "几", "最", "最多", "最低"}:
            break
        if word in {"，", ",", "？", "?", "。", ".", "的"}:
            break
        if word in dimension_words or is_time_token(word, analysis.pos[index]):
            index += 1
            continue
        compound = local_locative_compound_at(analysis, index)
        if compound is not None:
            end, value = compound
            if not any(analysis.words[position] in dimension_words for position in range(index, end + 1)):
                selected.append(value)
            index = end + 1
            continue
        if analysis.pos[index].startswith("N") or word in NOMINAL_COMPOUND_PARTS:
            selected.append(word)
            index += 1
            continue
        break
    return normalize_entity_suffix("".join(selected))


def strip_question_scaffold(value: str) -> str:
    """删除落入“X 的 Y”左侧的问句框架，不删除真实业务条件。

    HanLP重排后常见“业务有多少个是由于…导致的项目”。其中“有多少个是”
    是询问方式，“由于…导致”才是定语。该函数只在定语恢复入口使用，
    不影响原始句、查询意图或指标识别。
    """
    value = re.sub(r"^(?:运营商)?业务", "", value)
    value = re.sub(
        r"^(?:是否存在|是否有|有没有|有无|"
        r"(?:发生了?|还有|共有|总共有|有)?多少(?:个|张|起|次|项|例)?|"
        r"有哪些|哪个|哪些|哪几个|哪)+",
        "",
        value,
    )
    # “多少个是由于…”中的“是”是系词残片；保留“由/由于”，使因果条件
    # 仍能读作“由供方问题导致”或“由于物料供应问题导致”。
    return re.sub(r"^是", "", value)


def de_clause_condition_relations(
    analysis: object, dimension_words: set[str]
) -> list[tuple[str, str]]:
    """恢复“事件/状态短语 + 的 + 实体”，但排除普通范围名词短语。

    触发词限定为达成、保障、相关、合并、比拼等可独立表达业务条件的
    事件/关联词，故“哪个场景的项目”“业务的成本率”不会命中。
    """
    relations: list[tuple[str, str]] = []
    punctuation = {"，", ",", "。", ".", "？", "?", "；", ";"}
    for de_index, word in enumerate(analysis.words):
        if word != "的":
            continue
        start = de_index - 1
        while start >= 0 and analysis.words[start] not in punctuation and analysis.words[start] != "的":
            start -= 1
        left_indices = [
            index
            for index in range(start + 1, de_index)
            if analysis.words[index] not in dimension_words
            and not is_time_token(analysis.words[index], analysis.pos[index])
        ]
        if not left_indices:
            continue
        modifier = "".join(analysis.words[index] for index in left_indices)
        # 查询意图只描述问法，不能并入“X的Y”左侧业务条件。
        modifier = strip_question_scaffold(modifier)
        for prefix in GENERIC_OBJECT_PREFIXES:
            if modifier.startswith(prefix) and len(modifier) > len(prefix):
                modifier = modifier[len(prefix) :]
                break
        has_trigger = any(trigger in modifier for trigger in DE_CLAUSE_TRIGGERS)
        # “国家三领先达成有风险的国家”：风险本身只在有明确达成动作时
        # 扩展为完整事件条件，避免把所有“风险项目”误判为动词性短语。
        has_achievement_risk = "达成" in modifier and "风险" in modifier
        if not modifier or not (has_trigger or has_achievement_risk):
            continue
        entity = de_clause_entity(analysis, de_index, dimension_words)
        if entity != "—":
            identifiers = [
                analysis.words[index]
                for index in left_indices
                if is_business_identifier(analysis.words[index])
            ]
            # “FBB场景相关的交付风险项目”中，FBB和“场景相关”是两项
            # 独立条件；仅对“标识 + 相关”结构拆分，避免把“NPX保障”这类
            # 不可拆的业务名称误拆开。
            if identifiers and "相关" in modifier:
                # “FBB相关”是一个不可再拆的完整条件；若还有“场景”等
                # 实质片段（FBB场景相关），才分解为标识词和剩余条件。
                remainder = modifier
                for identifier in identifiers:
                    remainder = remainder.replace(identifier, "", 1)
                if remainder == "相关":
                    relations.append((modifier, entity))
                    continue
                for identifier in identifiers:
                    relations.append((identifier, entity))
                if remainder:
                    relations.append((remainder, entity))
            else:
                relations.append((modifier, entity))
    return relations


def prune_dominated_entity_relations(relations: list[str]) -> list[str]:
    """删除“高风险交付 → 项目”式、已被完整关系覆盖的中间链。"""
    pairs = [tuple(relation.split(" → ", 1)) for relation in relations]
    kept: list[str] = []
    for index, (modifier, entity) in enumerate(pairs):
        dominated = False
        for other_index, (short_modifier, full_entity) in enumerate(pairs):
            if index == other_index:
                continue
            # 当前关系是短条件、另一关系以它结尾且实体相同时，保留信息更全的
            # 条件。例如删除“有风险 → 国家”，保留“国家三领先达成有风险 → 国家”。
            if (
                # 标识词与“标识词相关”同时存在时，后者语义更完整；但
                # “FBB场景相关”不删除独立的 FBB 条件。
                is_business_identifier(modifier)
                and short_modifier == f"{modifier}相关"
                and full_entity == entity
            ):
                dominated = True
                break
            if (
                is_semantic_condition(modifier)
                and short_modifier.endswith(modifier)
                and short_modifier != modifier
                and full_entity == entity
            ):
                dominated = True
                break
            if not is_semantic_condition(short_modifier):
                continue
            if (
                not (modifier.startswith(short_modifier) or modifier.endswith(short_modifier))
                or modifier == short_modifier
            ):
                continue
            if not full_entity.endswith(entity) or full_entity == entity:
                continue
            # 只有左右两侧移出的片段相同才视为同一条链，避免误删普通名词链。
            moved_from_modifier = modifier[len(short_modifier) :]
            moved_from_entity = full_entity[: len(full_entity) - len(entity)]
            if moved_from_modifier and moved_from_modifier == moved_from_entity:
                dominated = True
                break
        if not dominated:
            kept.append(relations[index])
    return kept


def split_coordinated_entity_relations(relations: list[str]) -> list[str]:
    """受控展开并列定语，保留普通业务实体内部的并列结构。"""
    expanded: list[str] = []
    seen: set[str] = set()

    def independently_filterable(value: str) -> bool:
        return is_semantic_condition(value) or any(
            trigger in value for trigger in DE_CLAUSE_TRIGGERS
        )

    for relation in relations:
        modifier, entity = relation.split(" → ", 1)
        state_prefix = ""
        coordinated_value = modifier
        # “存在高风险或中风险”共享“存在”，展开时复制给每一支。
        for prefix in ("存在", "有", "无"):
            if modifier.startswith(prefix) and len(modifier) > len(prefix):
                state_prefix = prefix
                coordinated_value = modifier[len(prefix) :]
                break
        parts = [part for part in re.split(r"[和或及、]", coordinated_value) if part]
        candidates = [state_prefix + part for part in parts]
        if len(parts) > 1 and all(independently_filterable(item) for item in candidates):
            for item in candidates:
                split_relation = f"{item} → {entity}"
                if split_relation not in seen:
                    seen.add(split_relation)
                    expanded.append(split_relation)
            continue
        if relation not in seen:
            seen.add(relation)
            expanded.append(relation)
    return expanded


def expanded_entity_phrase(
    analysis: object, head_index: int, blocked_words: set[str] | None = None
) -> str:
    """以中心名词为锚点恢复完整业务实体，补偿“交付/比拼”误作谓词。"""
    allowed = {"nn", "amod", "ordmod", "clf", "conj", "cc"}
    selected = {head_index}
    frontier = [head_index]
    while frontier:
        parent = frontier.pop()
        for child in children(analysis, parent, allowed):
            selected.add(child)
            frontier.append(child)

    for child in children(analysis, head_index, {"rcmod", "vmod"}):
        if not is_expandable_compound_child(analysis, child, head_index):
            continue
        selected.add(child)
        for argument in children(analysis, child, {"nn", "amod", "ordmod", "nsubj", "dobj"}):
            if analysis.pos[argument].startswith("N") or analysis.labels[argument] in {"nn", "amod", "ordmod"}:
                selected.update(static_subtree_indices(analysis, argument))
                selected.add(argument)
        for modifier in children(analysis, child, {"advmod"}):
            if analysis.words[modifier] in RISK_WORDS:
                selected.add(modifier)

    selected = enrich_local_locative_compounds(analysis, selected)
    blocked_words = blocked_words or set()
    excluded = {"多少", "几", "个", "起", "次", "张"}
    value = "".join(
        analysis.words[index]
        for index in sorted(selected)
        if analysis.words[index] not in blocked_words | excluded
        and not is_entity_condition_component(analysis.words[index], analysis.labels[index])
        and not (
            analysis.labels[index] == "clf"
            and analysis.words[index] == "级"
            and bool(children(analysis, index, {"ordmod"}))
        )
        and analysis.words[index] not in COORDINATION_MARKERS
        and not is_time_token(analysis.words[index], analysis.pos[index])
    )
    return normalize_entity_suffix(value)


def postposed_adjective_state(
    analysis: object, predicate_index: int, dimension_words: set[str]
) -> tuple[str, str] | None:
    """恢复“网络质量不健康”一类句末否定状态。"""
    if not analysis.pos[predicate_index].startswith("V"):
        return None
    negation = "".join(analysis.words[index] for index in children(analysis, predicate_index, {"neg"}))
    if not negation:
        return None
    subjects = children(analysis, predicate_index, {"nsubj"})
    if not subjects:
        return None
    entity = expanded_entity_phrase(analysis, subjects[0], dimension_words)
    if entity == "—":
        return None
    return f"{negation}{analysis.words[predicate_index]}", entity


def modifier_entity_relations(analysis: object, dimension_words: set[str]) -> str:
    """输出纯名词定中链和已恢复动词性定语的“修饰→实体”候选。"""
    relations: list[str] = []
    seen: set[str] = set()

    def add(modifier: str, entity: str) -> None:
        relation = f"{modifier} → {entity}"
        # 未合并分词时的“高/中/低 → 风险”只保留在原子DEP证据中，
        # 不作为面向RAG的最终业务条件。
        is_fragmented_risk_level = modifier in RISK_LEVEL_PARTS and entity == "风险"
        # “交付项目/交付场景”是业务实体本体；“交付 → 项目/场景”只是
        # 内部组成，不能作为RAG筛选条件。
        is_non_condition_compound = modifier == "交付" and entity in {"项目", "场景"}
        if (
            modifier != "—"
            and entity != "—"
            and modifier != entity
            and not is_fragmented_risk_level
            and not is_non_condition_compound
            and relation not in seen
        ):
            seen.add(relation)
            relations.append(relation)

    # 静态定中：最终列只接纳等级/风险等语义条件，不输出普通名词组成链。
    # 原子 DEP 列仍完整保留“交付 -[nn]→ 项目”等句法证据。
    static_labels = {"nn", "amod", "assmod"}
    for head_index, head_label in enumerate(analysis.labels):
        if head_label not in {"dobj", "top", "lobj", "pobj"} or analysis.pos[head_index] == "NT":
            continue
        child_indices = sorted(children(analysis, head_index, static_labels))
        # 介词范围只处理“这/该/这些/上述 + 复合实体”的指代上下文，
        # 例如“在这4个S级交付项目中”；普通背景范围不当作实体修饰展开。
        if head_label in {"lobj", "pobj"}:
            has_reference = any(
                analysis.words[index] in {"这", "该", "这些", "上述"}
                for index in children(analysis, head_index, {"det"})
            )
            if len(child_indices) < 2 or not has_reference:
                continue
        component_entries: list[tuple[int, str]] = []
        for child in child_indices:
            subtree = static_subtree_indices(analysis, child)
            if any(analysis.words[index] in dimension_words for index in subtree):
                continue
            component = coordinated_static_component(analysis, child, dimension_words)
            if component:
                component_entries.append((child, component))
        components = merge_coordinated_components(analysis, component_entries)
        entity = expanded_entity_phrase(analysis, head_index, dimension_words)
        if entity == "—":
            continue
        for component in components:
            if is_semantic_condition(component):
                add(component, entity)
            elif is_business_identifier(component):
                # 标识词是实体外部限定语，不能被并入右侧实体。
                identifier_free_entity = expanded_entity_phrase(
                    analysis, head_index, dimension_words | {component}
                )
                add(component, identifier_free_entity)

    # 等级短语常被切为“一 / 级”且DEP边缺失，单独从连续词序恢复。
    # 例如“一级和二级网上事故”恢复为“一级和二级 → 网上事故”。
    for modifier, entity in level_condition_relations(analysis, dimension_words):
        add(modifier, entity)

    # 否定状态型rcmod不必局限于预设动词表，例如“恢复不及时的网上事故”。
    for predicate in range(len(analysis.words)):
        modifier = negative_rcmod_state(analysis, predicate)
        if modifier is None or analysis.heads[predicate] == 0:
            continue
        entity = expanded_entity_phrase(
            analysis, analysis.heads[predicate] - 1, dimension_words
        )
        add(modifier, entity)

    # 对DEP断裂更敏感的“事件/状态短语 + 的 + 实体”使用局部词序补偿。
    for modifier, entity in de_clause_condition_relations(analysis, dimension_words):
        add(modifier, entity)

    # 统计后缀不是定语中心词；但其左侧业务对象中的 amod/ordmod 仍是
    # 有价值的条件。例如“高风险客户数”只输出“高风险 → 客户”。
    for modifier, entity in statistical_metric_entity_candidates(analysis, dimension_words):
        add(modifier, entity)

    # 复合名词断裂补偿：仅接受“无的、无动词论元”的 bare rcmod，
    # 并要求它有左侧 dep/nn/amod 名词性片段。此时将 rcmod 词本身并入实体。
    for predicate, label in enumerate(analysis.labels):
        if label != "rcmod" or analysis.heads[predicate] == 0:
            continue
        head_index = analysis.heads[predicate] - 1
        prefix = bare_rcmod_prefix(analysis, predicate, head_index, dimension_words)
        if prefix is None:
            continue
        base_entity = expanded_entity_phrase(analysis, head_index, dimension_words)
        if base_entity == "—":
            continue
        tail = normalize_entity_suffix(
            f"{analysis.words[predicate]}{base_entity}"
        )
        if is_semantic_condition(prefix):
            add(prefix, tail)

    # “交付风险项目的TOP3原因”一类结构：名词被误标为rcmod时，
    # 用“的”及其局部名词子树恢复完整的左、右两侧短语。
    for predicate, label in enumerate(analysis.labels):
        if label not in {"rcmod", "vmod"} or analysis.heads[predicate] == 0:
            continue
        candidate = nominal_rcmod_de_candidate(analysis, predicate, dimension_words)
        if candidate is not None:
            add(*candidate)

    # “比拼/交付”在数量问句中常为复合实体成分，却被DEP标为动词。
    for predicate in range(len(analysis.words)):
        if (
            analysis.labels[predicate] in {"rcmod", "vmod"}
            and analysis.heads[predicate] != 0
            and is_metric_head(analysis.words[analysis.heads[predicate] - 1])
        ):
            continue
        components = nominalized_compound_components(analysis, predicate, dimension_words)
        if components is None:
            continue
        # 复合名词本身不是ATT；仅当其首段是等级/风险等条件时，
        # 输出该条件到其余完整实体，例如“高风险 → 交付售前项目”。
        if len(components) > 1 and (
            is_semantic_condition(components[0]) or is_business_identifier(components[0])
        ):
            add(components[0], "".join(components[1:]))

    # 风险等级本身是完整业务条件。即使DEP把“高风险”误作 rcmod，或把
    # “高风险/中风险”并列关系标错，也保留它到局部业务实体的完整关系。
    for index, word in enumerate(analysis.words):
        if word not in RISK_WORDS or analysis.heads[index] == 0:
            continue
        # 静态 amod/nn 已由上方定中链输出；这里只补rcmod/vmod，以及
        # 挂在rcmod/vmod上的并列风险词（重大或高风险）。
        predicate = attached_att_predicate(analysis, index)
        if predicate is None:
            continue
        # “存在高风险的项目”已有“存在高风险 → 项目”的完整状态关系；
        # 不额外降级输出“高风险 → 项目”。
        if analysis.words[predicate] == "存在":
            continue
        entity = entity_after_risk_condition(
            analysis, index, analysis.heads[predicate] - 1, dimension_words
        )
        add(risk_condition_around(analysis, index), entity)

    # 无“的”的句末状态没有 rcmod 边，单独以“存在 + 完整风险条件”输出。
    for predicate, word in enumerate(analysis.words):
        if word == "存在":
            state = postposed_risk_state(analysis, predicate, dimension_words)
            if state is not None:
                add(*state)

    # 句末“X不健康”类状态谓词不依赖rcmod标签恢复。
    for predicate in range(len(analysis.words)):
        state = postposed_adjective_state(analysis, predicate, dimension_words)
        if state is not None:
            add(*state)
            subjects = children(analysis, predicate, {"nsubj"})
            if subjects:
                subject = subjects[0]
                for child in children(analysis, subject, {"amod"}):
                    modifier = nominal_phrase(analysis, child, dimension_words)
                    entity = static_entity_without(
                        analysis, subject, static_subtree_indices(analysis, child), dimension_words
                    )
                    add(modifier, entity)

    # 动词性定语：恢复为完整条件，并把中心词扩展为局部业务实体短语。
    for predicate, label in enumerate(analysis.labels):
        if label not in {"rcmod", "vmod"} or analysis.heads[predicate] == 0:
            continue
        predicate_word = analysis.words[predicate]
        if predicate_word not in RECOVERABLE_PREDICATES:
            continue
        modifiers = "".join(
            analysis.words[index]
            for index in sorted(children(analysis, predicate, PREDICATE_MODIFIER_LABELS))
            if analysis.words[index] not in NON_CORE_PREDICATE_MODIFIERS
        )
        if predicate_word in {"导致", "造成", "引起", "产生"}:
            modifier = marked_cause(analysis, predicate, dimension_words)
            if modifier is None:
                continue
        elif predicate_word in {"存在", "有", "无"}:
            objects = children(analysis, predicate, {"dobj"})
            condition = nominal_phrase(analysis, objects[0], dimension_words) if objects else ""
            modifier = f"{predicate_word}{condition}"
        else:
            modifier = f"{modifiers}{predicate_word}"
        entity = expanded_entity_phrase(analysis, analysis.heads[predicate] - 1, dimension_words)
        add(modifier, entity)
    relations = split_coordinated_entity_relations(relations)
    return "；".join(prune_dominated_entity_relations(relations)) or "—"


def decisions_for(
    records: list[tuple[int, str]], plans: list[object], candidate_analyses: dict[int, object]
) -> list[Decision]:
    decisions: list[Decision] = []
    for index, ((_line_number, sentence), plan) in enumerate(zip(records, plans, strict=True)):
        if plan is None:
            decisions.append(Decision("not_applicable", sentence, "未命中安全重排模式", "—"))
        else:
            decisions.append(validate_plan(sentence, plan, candidate_analyses[index], "hanlp"))
    return decisions


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    original_analyses = load_analyses(
        [sentence for _line_number, sentence in records], args.model_home, args.batch_size
    )
    dimension_lines = read_dimension_lines(args.dimension_input)
    if len(dimension_lines) > len(records):
        raise ValueError(
            f"维度文件行数异常：{len(dimension_lines)}条，原始问题为{len(records)}条"
        )
    dimension_words_by_row = {
        line_number: removed_dimension_words(
            sentence,
            dimension_lines[line_number - 2] if line_number - 2 < len(dimension_lines) else "",
            analysis.words,
        )
        for (line_number, sentence), analysis in zip(records, original_analyses, strict=True)
    }
    # 维度提取结果会删除地域、组织等范围，也可能误删FBB、NPX、P3等
    # 业务限定标识。后者应继续参与实体条件抽取，而不是当作结构维度过滤。
    dimension_words_by_row = {
        line_number: {
            word for word in words if not is_business_identifier(word)
        }
        for line_number, words in dimension_words_by_row.items()
    }
    plans = [make_plan(analysis, "hanlp") for analysis in original_analyses]
    candidate_indices, candidates = candidate_sentences(plans)
    candidate_values = load_analyses(candidates, args.model_home, args.batch_size)
    candidate_analyses = dict(zip(candidate_indices, candidate_values, strict=True))
    decisions = decisions_for(records, plans, candidate_analyses)

    rows: list[tuple[str, ...]] = []
    for index, ((line_number, original), decision, original_analysis) in enumerate(
        zip(records, decisions, original_analyses, strict=True)
    ):
        if decision.status == "accepted":
            final_sentence = decision.output
            final_analysis = candidate_analyses[index]
            reordered = final_sentence
        else:
            final_analysis = original_analysis
            reordered = "—"
        dimension_sentence = (
            dimension_lines[line_number - 2] if line_number - 2 < len(dimension_lines) else ""
        )
        # 后续五列的语义均是“去时间、维度后”的派生结果。若维度提取句
        # 为空，则没有可验证的过滤基准；不能一边展示“—”，一边把原句的
        # DEP 结果混入这些列。
        if dimension_sentence:
            dimensionless = dimensionless_final_sentence(
                final_analysis, dimension_words_by_row[line_number], True
            )
            relations = att_relations(final_analysis, dimension_words_by_row[line_number])
            final = modifier_entity_relations(final_analysis, dimension_words_by_row[line_number])
            metric = object_metric_relations(final_analysis, dimension_words_by_row[line_number])
            # DEP 有时把条件一并吞入对象（如“未关闭二级管理升级 → 数量”）。
            # 已验证的“条件 → 实体”提供边界证据，据此让对象回到主实体。
            metric = normalize_object_metric_with_conditions(metric, final)
            final = normalize_condition_entities_with_metric(final, metric)
            intent = query_intent(final_analysis, dimension_words_by_row[line_number])
            detail_entity = (
                detail_query_entity_candidate(final_analysis, dimension_words_by_row[line_number])
                if intent == "详情查询"
                else "—"
            )
            if detail_entity != "—":
                entities = detail_entity
            elif metric == "—" and intent == "数量查询":
                condition_entities = [
                    entity
                    for _modifier, entity in relation_pairs(final)
                    if not entity.endswith(tuple(INDICATOR_TAILS))
                ]
                count_entity = count_question_entity_candidate(
                    final_analysis, dimension_words_by_row[line_number]
                )
                # 数量词局部结构比名词短语回退可靠；但若已经有“条件 → 实体”，
                # 则其右侧实体是更准确的中心词。
                entities = (
                    "；".join(dict.fromkeys(condition_entities))
                    if condition_entities
                    else count_entity
                )
                if entities == "—":
                    entities = business_entity_candidates(
                        final_analysis, dimension_words_by_row[line_number], final, metric
                    )
            else:
                entities = business_entity_candidates(
                    final_analysis, dimension_words_by_row[line_number], final, metric
                )
            # “哪个场景的风险项目最多”省略了“数量/项目数”，但“最多/最少”
            # 明确以对象个数排序。仅在实体候选唯一时补指标，避免把DEP拆开的
            # “变更；操作”之类歧义片段误当完整对象。
            if (
                metric == "—"
                and entities != "—"
                and "；" not in entities
                and has_implicit_count_extremum(final_analysis)
            ):
                metric = f"{entities} → 数量"
            # “有多少个高风险交付项目”没有显式指标尾部，但问法明确要求
            # 对主实体计数；补成通用指标“数量”，避免实体有而指标为空。
            if metric == "—" and intent == "数量查询" and entities != "—":
                primary_entity = entities.split("；", 1)[0]
                metric = f"{primary_entity} → 数量"
            indicators = indicator_candidates(metric)
        else:
            dimensionless = relations = final = metric = intent = entities = indicators = "—"
        rows.append(
            (
                str(line_number),
                original,
                " / ".join(final_analysis.words),
                dimensionless,
                reordered,
                intent,
                entities,
                indicators,
                relations,
                final,
                metric,
            )
        )

    with_entities = sum(row[6] != "—" for row in rows)
    with_indicators = sum(row[7] != "—" for row in rows)
    with_relations = sum(row[8] != "—" for row in rows)
    with_final = sum(row[9] != "—" for row in rows)
    with_metrics = sum(row[10] != "—" for row in rows)
    accepted = sum(decision.status == "accepted" for decision in decisions)
    lines = [
        "# HanLP DEP 去时间、维度后的原子定中关系",
        "",
        f"- 来源：`{args.input.as_posix()}`。",
        "- 流程：原始问题 → 受控语序重排及二次DEP+SRL校验 → 最终句子的TOK+POS+DEP。",
        "- 重排成功时采用重排句；未命中或校验失败时保留原句。",
        "- 关系范围：`nn`、`amod`、`assmod`、`rcmod`、`vmod`；DEP原子ATT列已过滤时间和维度端点。",
        "- 修饰→实体：结合动词性定语、否定状态、因果原因与bare rcmod局部证据恢复完整业务条件。",
        f"- 维度过滤：使用`{args.dimension_input.as_posix()}`逐行差异识别维度；对应维度提取句为空或缺失时，从“去时间、维度的问题”起的派生列统一输出`—`。指标和查询意图保留。",
        f"- 实体/指标候选：在最终句上去时间、维度和查询意图后生成；实体优先取已验证关系的对象/右侧实体，缺失时才回退局部名词短语。",
        f"- 统计：{len(rows)}条问题，自动接受重排{accepted}条，{with_entities}条生成实体候选，{with_indicators}条生成指标候选，{with_relations}条含至少一条定中关系，{with_final}条生成修饰→实体，{with_metrics}条生成对象→指标。",
        "",
        "| 原文件行号 | 原始问题 | 语序重排（如有） | 分词结果（最终句） | 去时间、维度的问题 | 查询意图 | 业务实体候选 | 指标候选 | 去时间、维度后的 DEP 原子ATT关系 | 修饰 → 实体 | 对象 → 指标 |",
        "|---:|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (
        line_number,
        original,
        tokens,
        dimensionless,
        reordered,
        intent,
        entities,
        indicators,
        relations,
        final,
        metric,
    ) in rows:
        lines.append(
            "| "
            + " | ".join(
                escape(value)
                for value in (
                    line_number,
                    original,
                    reordered,
                    tokens,
                    dimensionless,
                    intent,
                    entities,
                    indicators,
                    relations,
                    final,
                    metric,
                )
            )
            + " |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"DEP原子ATT结果已写入：{args.output}（{len(rows)}条）")


if __name__ == "__main__":
    main()
