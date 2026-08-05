from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CAUSE_MARKERS = {"由", "由于", "因为"}
CAUSE_PREDICATES = {"导致", "造成", "引起", "产生"}
CLASSIFIERS = {"个", "项", "起", "次", "例"}
QUANTIFIERS = {"多少", "几", "几个", "哪几", "哪几个"}
INTERROGATIVES = {"哪些", "哪个", "哪项", "哪几个"}
PUNCTUATION = {"，", ",", "。", ".", "？", "?", "！", "!", "；", ";"}


@dataclass
class Analysis:
    words: list[str]
    pos: list[str]
    heads: list[int]
    labels: list[str]
    srl: list[Any]
    predicates: list[int]
    argument_spans: dict[int, list[tuple[int, int, str]]]
    constituency: str = ""


@dataclass
class ReorderPlan:
    original_tokens: list[str]
    candidate_tokens: list[str]
    # None represents the only allowed inserted token: structural particle “的”.
    mapping: list[int | None]
    object_indices: list[int]
    object_head: int
    cause_predicate: int
    copula: int
    reason: str
    rule_type: str

    @property
    def candidate(self) -> str:
        return "".join(self.candidate_tokens)

    def candidate_index(self, original_index: int) -> int:
        return self.mapping.index(original_index)


@dataclass
class Decision:
    status: str
    output: str
    reason: str
    evidence: str


def read_records(path: Path) -> list[tuple[int, str]]:
    return [
        (line_number, line.strip())
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line_number > 1 and line.strip()
    ]


def _find_object_start(words: list[str], copula: int) -> int | None:
    # Do not borrow a quantity phrase from a previous clause or question.
    lower_bound = 0
    for index in range(copula - 1, -1, -1):
        if words[index] in PUNCTUATION:
            lower_bound = index + 1
            break
    for index in range(copula - 1, lower_bound - 1, -1):
        if words[index] not in CLASSIFIERS:
            continue
        left = words[max(0, index - 2) : index]
        if any(word in QUANTIFIERS for word in left):
            return index + 1
    for index in range(copula - 1, lower_bound - 1, -1):
        if words[index] in INTERROGATIVES:
            return index + 1
    return None


def _find_quantity_object_start(words: list[str], predicate: int) -> int | None:
    """Find an object after a quantified classifier; do not match 哪个/哪些 questions."""
    lower_bound = 0
    for index in range(predicate - 1, -1, -1):
        if words[index] in PUNCTUATION:
            lower_bound = index + 1
            break
    for index in range(predicate - 1, lower_bound - 1, -1):
        if words[index] not in CLASSIFIERS:
            continue
        if any(word in QUANTIFIERS for word in words[max(lower_bound, index - 2) : index]):
            return index + 1
    return None


def _is_noun(tag: str, backend: str) -> bool:
    if backend == "ltp":
        return tag.startswith("n")
    return tag.startswith("N")


def _object_head(analysis: Analysis, object_indices: list[int], backend: str) -> int:
    object_set = set(object_indices)
    roots = [
        index
        for index in object_indices
        if analysis.heads[index] == 0 or analysis.heads[index] - 1 not in object_set
    ]
    noun_roots = [index for index in roots if _is_noun(analysis.pos[index], backend)]
    return (noun_roots or roots or object_indices)[-1]


def _sentence_end(words: list[str], start: int) -> int:
    """Return the first punctuation token at or after start, or sentence length."""
    for index in range(start, len(words)):
        if words[index] in PUNCTUATION:
            return index
    return len(words)


def make_causal_plan(analysis: Analysis, backend: str) -> ReorderPlan | None:
    """Detect only the safe postposed causal-attributive pattern.

    Supported shape: ``...多少个 + OBJECT + 是由于/由/因为 + CAUSE + 导致的``.
    The generated sentence is a permutation of the original tokens.
    """
    words = analysis.words
    for copula, word in enumerate(words):
        if word != "是":
            continue
        marker = next(
            (
                index
                for index in range(copula + 1, min(len(words), copula + 4))
                if words[index] in CAUSE_MARKERS
            ),
            None,
        )
        if marker is None:
            continue
        predicate = next(
            (
                index
                for index in range(marker + 1, len(words))
                if words[index] in CAUSE_PREDICATES
            ),
            None,
        )
        if predicate is None or predicate + 1 >= len(words) or words[predicate + 1] != "的":
            continue
        clause_end = predicate + 2
        if any(token not in PUNCTUATION for token in words[clause_end:]):
            continue
        object_start = _find_object_start(words, copula)
        if object_start is None or object_start >= copula:
            continue
        object_indices = list(range(object_start, copula))
        # Nominalized business terms are often tagged as verbs (e.g. 管理升级、变更倒回).
        # Prefer the dependency root inside the object span instead of rejecting by POS.
        object_head = _object_head(analysis, object_indices, backend)
        if predicate not in analysis.predicates:
            continue

        mapping = (
            list(range(object_start))
            + list(range(copula, clause_end))
            + object_indices
            + list(range(clause_end, len(words)))
        )
        candidate_tokens = [words[index] for index in mapping]
        if sorted(mapping) != list(range(len(words))):
            raise AssertionError("重排映射不是原token下标的全排列")
        if "".join(candidate_tokens) != (
            "".join(words[:object_start])
            + "".join(words[copula:clause_end])
            + "".join(words[object_start:copula])
            + "".join(words[clause_end:])
        ):
            raise AssertionError("候选句构造失败")
        return ReorderPlan(
            original_tokens=words,
            candidate_tokens=candidate_tokens,
            mapping=mapping,
            object_indices=object_indices,
            object_head=object_head,
            cause_predicate=predicate,
            copula=copula,
            reason="识别到‘数量短语+业务对象+是由/由于/因为…导致的’后置因果结构",
            rule_type="因果型：后置因果定语前置",
        )
    return None


def _find_initial_location_end(analysis: Analysis, lower_bound: int, has: int, backend: str) -> int:
    """Keep an initial place name before 有; the remaining span is the business entity.

    This is deliberately conservative: only a leading location POS is separated.
    Other modifiers remain part of the entity span.
    """
    location_tags = {"ns"} if backend == "ltp" else {"NR"}
    index = lower_bound
    while index < has and analysis.pos[index] in location_tags:
        index += 1
    return index


def make_existence_plan(analysis: Analysis, backend: str) -> ReorderPlan | None:
    """Rewrite postposed existential conditions, adding only the particle “的”.

    Direct form: ``多少个 + OBJECT + 存在 + CONDITION``
    Scope form: ``OBJECT（中）有多少个存在 + CONDITION``
    """
    words = analysis.words
    for predicate, word in enumerate(words):
        if word != "存在" or predicate not in analysis.predicates:
            continue
        clause_end = _sentence_end(words, predicate)
        condition = list(range(predicate + 1, clause_end))
        if not condition:
            continue
        object_start = _find_quantity_object_start(words, predicate)
        if object_start is not None and object_start < predicate:
            object_indices = list(range(object_start, predicate))
            mapping: list[int | None] = (
                list(range(object_start))
                + [predicate]
                + condition
                + [None]
                + object_indices
                + list(range(clause_end, len(words)))
            )
            candidate_tokens = (
                words[:object_start]
                + [words[predicate]]
                + [words[index] for index in condition]
                + ["的"]
                + [words[index] for index in object_indices]
                + words[clause_end:]
            )
            return ReorderPlan(
                original_tokens=words,
                candidate_tokens=candidate_tokens,
                mapping=mapping,
                object_indices=object_indices,
                object_head=_object_head(analysis, object_indices, backend),
                cause_predicate=predicate,
                copula=-1,
                reason="识别到‘数量短语+业务对象+存在+条件’后置存在型条件",
                rule_type="存在型：后置条件前置（新增“的”）",
            )

        # Scope form: OBJECT（中）有多少个存在 CONDITION.
        classifier = next(
            (
                index
                for index in range(predicate - 1, -1, -1)
                if words[index] in CLASSIFIERS
                and any(token in QUANTIFIERS for token in words[max(0, index - 2) : index])
            ),
            None,
        )
        if classifier is None:
            continue
        has = next(
            (index for index in range(classifier - 1, -1, -1) if words[index] == "有"),
            None,
        )
        if has is None or has >= classifier:
            continue
        lower_bound = 0
        for index in range(has - 1, -1, -1):
            if words[index] in PUNCTUATION:
                lower_bound = index + 1
                break
        entity_start = _find_initial_location_end(analysis, lower_bound, has, backend)
        entity_indices = list(range(entity_start, has))
        if entity_indices and words[entity_indices[-1]] == "中":
            entity_indices.pop()
        if not entity_indices:
            continue
        mapping = (
            list(range(entity_start))
            + list(range(has, predicate))
            + [predicate]
            + condition
            + [None]
            + entity_indices
            + list(range(clause_end, len(words)))
        )
        candidate_tokens = (
            words[:entity_start]
            + words[has:predicate]
            + [words[predicate]]
            + [words[index] for index in condition]
            + ["的"]
            + [words[index] for index in entity_indices]
            + words[clause_end:]
        )
        return ReorderPlan(
            original_tokens=words,
            candidate_tokens=candidate_tokens,
            mapping=mapping,
            object_indices=entity_indices,
            object_head=_object_head(analysis, entity_indices, backend),
            cause_predicate=predicate,
            copula=has,
            reason="识别到‘业务对象（中）有多少个存在+条件’范围结构",
            rule_type="存在型：范围结构前置（新增“的”；可去除“中”）",
        )
    return None


def make_state_plan(analysis: Analysis, backend: str) -> ReorderPlan | None:
    """Rewrite postposed states: OBJECT + 未关闭/未恢复/还未恢复."""
    words = analysis.words
    for predicate, word in enumerate(words):
        if word not in {"关闭", "恢复"} or predicate not in analysis.predicates:
            continue
        if predicate == 0 or words[predicate - 1] != "未":
            continue
        state_start = predicate - 1
        if state_start > 0 and words[state_start - 1] == "还":
            state_start -= 1
        clause_end = _sentence_end(words, predicate)
        # This first version accepts only closed state phrases, not arbitrary verb objects.
        if clause_end != predicate + 1:
            continue
        object_start = _find_quantity_object_start(words, state_start)
        if object_start is None or object_start >= state_start:
            continue
        object_indices = list(range(object_start, state_start))
        mapping: list[int | None] = (
            list(range(object_start))
            + list(range(state_start, predicate + 1))
            + [None]
            + object_indices
            + list(range(clause_end, len(words)))
        )
        candidate_tokens = (
            words[:object_start]
            + words[state_start : predicate + 1]
            + ["的"]
            + [words[index] for index in object_indices]
            + words[clause_end:]
        )
        return ReorderPlan(
            original_tokens=words,
            candidate_tokens=candidate_tokens,
            mapping=mapping,
            object_indices=object_indices,
            object_head=_object_head(analysis, object_indices, backend),
            cause_predicate=predicate,
            copula=-1,
            reason="识别到‘数量短语+业务对象+未关闭/未恢复’后置状态结构",
            rule_type="状态型：后置状态前置（新增“的”）",
        )
    return None


def make_copular_state_plan(analysis: Analysis, backend: str) -> ReorderPlan | None:
    """将“多少个对象是有/无状态的”受控改写为前置状态定语。

    只删除系词“是”，其余业务词和数量/范围上下文均保留：
    “有多少个项目是有风险的”→“有多少个有风险的项目”。
    """
    words = analysis.words
    for copula, word in enumerate(words):
        if word != "是" or copula + 2 >= len(words):
            continue
        predicate = copula + 1
        if words[predicate] not in {"有", "无"} or predicate not in analysis.predicates:
            continue
        clause_end = _sentence_end(words, predicate)
        particle = next(
            (index for index in range(predicate + 1, clause_end) if words[index] == "的"),
            None,
        )
        if particle is None or particle + 1 != clause_end:
            continue
        object_start = _find_object_start(words, copula)
        if object_start is None or object_start >= copula:
            continue
        object_indices = list(range(object_start, copula))
        if not object_indices:
            continue
        state_indices = list(range(predicate, particle + 1))
        mapping = (
            list(range(object_start))
            + state_indices
            + object_indices
            + list(range(clause_end, len(words)))
        )
        candidate_tokens = (
            words[:object_start]
            + [words[index] for index in state_indices]
            + [words[index] for index in object_indices]
            + words[clause_end:]
        )
        return ReorderPlan(
            original_tokens=words,
            candidate_tokens=candidate_tokens,
            mapping=mapping,
            object_indices=object_indices,
            object_head=_object_head(analysis, object_indices, backend),
            cause_predicate=predicate,
            copula=copula,
            reason="识别到‘数量短语+业务对象+是+有/无状态+的’判断句",
            rule_type="判断状态型：有/无状态前置（删除“是”）",
        )
    return None


def make_plan(analysis: Analysis, backend: str) -> ReorderPlan | None:
    return (
        make_causal_plan(analysis, backend)
        or make_existence_plan(analysis, backend)
        or make_copular_state_plan(analysis, backend)
        or make_state_plan(analysis, backend)
    )


def _has_srl_object(analysis: Analysis, predicate: int, object_indices: set[int]) -> bool:
    return any(
        any(index in object_indices for index in range(start, end + 1))
        for start, end, _label in analysis.argument_spans.get(predicate, [])
    )


def _token_offsets(words: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        offsets.append((cursor, cursor + len(word)))
        cursor += len(word)
    return offsets


def _actual_index_at(offsets: list[tuple[int, int]], char_index: int) -> int:
    for index, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return index
    raise ValueError(f"找不到字符位置{char_index}对应的二次分词token")


def validate_plan(
    original_sentence: str,
    plan: ReorderPlan,
    candidate_analysis: Analysis,
    backend: str,
) -> Decision:
    if "".join(candidate_analysis.words) != plan.candidate:
        return Decision(
            "rejected",
            original_sentence,
            "重排后二次分词未覆盖完整候选句",
            "二次分词字符不一致",
        )
    mapped_original = "".join(
        plan.original_tokens[index] for index in plan.mapping if index is not None
    )
    inserted = "".join(
        token for token, index in zip(plan.candidate_tokens, plan.mapping, strict=True) if index is None
    )
    if Counter(plan.candidate) != Counter(mapped_original + inserted):
        return Decision(
            "rejected",
            original_sentence,
            "候选字符不等于允许移动、删除或新增的token集合",
            "字符守恒失败",
        )

    intended_offsets = _token_offsets(plan.candidate_tokens)
    actual_offsets = _token_offsets(candidate_analysis.words)
    intended_predicate = plan.candidate_index(plan.cause_predicate)
    predicate_start, _ = intended_offsets[intended_predicate]
    new_predicate = _actual_index_at(actual_offsets, predicate_start)
    modifier_labels = {"ATT"} if backend == "ltp" else {"rcmod", "vmod", "assmod"}
    intended_object_indices = [plan.candidate_index(index) for index in plan.object_indices]
    object_start = min(intended_offsets[index][0] for index in intended_object_indices)
    object_end = max(intended_offsets[index][1] for index in intended_object_indices)
    new_object_indices = {
        index
        for index, (start, end) in enumerate(actual_offsets)
        if start < object_end and end > object_start
    }
    predicate_head = candidate_analysis.heads[new_predicate] - 1
    intended_head = plan.candidate_index(plan.object_head)
    _, intended_head_end = intended_offsets[intended_head]
    actual_object_head = _actual_index_at(actual_offsets, intended_head_end - 1)
    dep_ok = (
        predicate_head in new_object_indices
        and candidate_analysis.labels[new_predicate] in modifier_labels
    )
    # For existence, the head must be the entity center. For a state phrase such
    # as “未关闭的管理升级单”, LTP may attach 关闭 to 管理 while SRL correctly
    # recovers the whole entity, so requiring the final token would be too strict.
    if plan.rule_type.startswith(("存在型", "判断状态型")):
        dep_ok = dep_ok and predicate_head == actual_object_head
    srl_ok = _has_srl_object(candidate_analysis, new_predicate, new_object_indices)
    # 部分HanLP因果候选只在SRL中保留原因论元（如“由于服务质量”），
    # 不重复标注已由DEP rcmod挂接的业务对象。对已命中严格因果模板的候选，
    # DEP对象挂接成立且SRL存在任一因果论元即可作为替代校验。
    if plan.rule_type.startswith("因果型") and not srl_ok:
        srl_ok = bool(candidate_analysis.argument_spans.get(new_predicate))

    actual_head = (
        candidate_analysis.words[predicate_head]
        if 0 <= predicate_head < len(candidate_analysis.words)
        else "ROOT"
    )
    con_note = ""
    if backend == "hanlp":
        con_ok = (
            "'CP'" in candidate_analysis.constituency
            and "'NP'" in candidate_analysis.constituency
        )
        con_note = (
            f"；CON={'支持' if con_ok else '未提供明确支持'}："
            f"候选中{'存在' if con_ok else '未识别'}CP/NP结构"
        )

    evidence = (
        f"DEP={'通过' if dep_ok else '未通过'}："
        f"{candidate_analysis.words[new_predicate]} -[{candidate_analysis.labels[new_predicate]}]→ "
        f"{actual_head}；"
        f"SRL={'通过' if srl_ok else '未通过'}：目标谓词"
        f"{'包含' if srl_ok else '未包含'}业务对象论元"
        f"{con_note}"
    )
    if dep_ok and srl_ok:
        return Decision("accepted", plan.candidate, "DEP与SRL二次解析均通过", evidence)
    return Decision("rejected", original_sentence, "二次句法校验未同时通过", evidence)


def format_dep(analysis: Analysis) -> str:
    items = []
    for index, (word, head, label) in enumerate(
        zip(analysis.words, analysis.heads, analysis.labels, strict=True), start=1
    ):
        head_word = "ROOT" if head == 0 else analysis.words[head - 1]
        items.append(f"{index}:{word} -[{label}]→ {head}:{head_word}")
    return "；".join(items) or "无"


def format_srl(analysis: Analysis) -> str:
    frames = []
    for predicate in analysis.predicates:
        arguments = "，".join(
            f"{label}={''.join(analysis.words[start:end + 1])}[{start + 1}-{end + 1}]"
            for start, end, label in analysis.argument_spans.get(predicate, [])
        ) or "无论元"
        frames.append(f"{analysis.words[predicate]}@{predicate + 1}（{arguments}）")
    return "；".join(frames) or "无"


def escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def write_markdown(
    output: Path,
    source: Path,
    backend: str,
    model_description: str,
    records: list[tuple[int, str]],
    original_analyses: list[Analysis],
    plans: list[ReorderPlan | None],
    candidate_analyses: dict[int, Analysis],
) -> tuple[int, int, list[Decision]]:
    decisions: list[Decision] = []
    for index, ((_line, sentence), plan) in enumerate(zip(records, plans, strict=True)):
        if plan is None:
            decisions.append(Decision("not_applicable", sentence, "未命中安全重排模式", "—"))
        else:
            decisions.append(
                validate_plan(sentence, plan, candidate_analyses[index], backend)
            )

    accepted = sum(decision.status == "accepted" for decision in decisions)
    candidates = sum(plan is not None for plan in plans)
    lines = [
        f"# {source.stem} {backend.upper()} 语序重排结果",
        "",
        f"- 来源：`{source.as_posix()}`。",
        f"- 模型：{model_description}。",
        "- 范围：处理后置因果、存在条件和“是有/无状态的”判断句。",
        "- 改写约束：因果型只移动原token；存在型只允许新增‘的’，范围结构允许同时去除末尾结构词‘中’；判断状态型只删除系词“是”。",
        "- 自动接受：重排后必须保持候选句分词完整，且DEP恢复动词性定语、SRL恢复业务对象或（因果型）原因论元。",
        f"- 统计：共{len(records)}句，生成{candidates}个候选，自动接受{accepted}个。",
        "- token映射：候选token位置对应的原句token序号（均从1开始）。",
        "",
        "| 原文件行号 | 原句 | 命中规则 | 原始分词/POS | 原始DEP | 原始SRL | 重排候选 | 候选DEP | 候选SRL | token映射 | 状态 | 最终句子 | 判定依据 |",
        "|---:|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for index, ((line_number, sentence), original, plan, decision) in enumerate(
        zip(records, original_analyses, plans, decisions, strict=True)
    ):
        candidate = plan.candidate if plan else "—"
        candidate_analysis = candidate_analyses.get(index)
        mapping = (
            " / ".join("新增‘的’" if item is None else str(item + 1) for item in plan.mapping)
            if plan
            else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(line_number),
                    escape(sentence),
                    escape(plan.rule_type if plan else "—"),
                    escape(" / ".join(f"{w}/{p}" for w, p in zip(original.words, original.pos, strict=True))),
                    escape(format_dep(original)),
                    escape(format_srl(original)),
                    escape(candidate),
                    escape(format_dep(candidate_analysis) if candidate_analysis else "—"),
                    escape(format_srl(candidate_analysis) if candidate_analysis else "—"),
                    escape(mapping),
                    decision.status,
                    escape(decision.output),
                    escape(f"{decision.reason}；{decision.evidence}"),
                ]
            )
            + " |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return candidates, accepted, decisions


def candidate_sentences(
    plans: Iterable[ReorderPlan | None],
) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    sentences: list[str] = []
    for index, plan in enumerate(plans):
        if plan is not None:
            indices.append(index)
            sentences.append(plan.candidate)
    return indices, sentences
