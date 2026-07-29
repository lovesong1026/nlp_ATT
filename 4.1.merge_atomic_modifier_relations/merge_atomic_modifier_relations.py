from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from typing import Any

from ltp import LTP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE4_SCRIPT = (
    PROJECT_ROOT
    / "4.extract_atomic_modifier_relations"
    / "extract_atomic_modifier_relations.py"
)
DEFAULT_INPUT = PROJECT_ROOT / "data/original_question.md"
DEFAULT_DIMENSION_INPUT = (
    PROJECT_ROOT / "data/dimension_extracted_question.md"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "original_question_merged_modifier_relations.md"
)
DEFAULT_DICTIONARY = PROJECT_ROOT / "segmentation_words.txt"

BRIDGE_WORDS = {
    "的",
    "之",
    "和",
    "与",
    "及",
    "或",
    "以及",
    "或者",
}
QUERY_WORDS = {
    "哪个",
    "哪些",
    "哪",
    "多少",
    "几",
    "每",
    "一个",
    "个",
    "次",
    "张",
    "项",
    "条",
}
GENERIC_VERBAL_MODIFIERS = {
    "导致的",
    "发生的",
    "涉及的",
    "相关的",
    "存在的",
    "有的",
}
ATTRIBUTE_POS = {"a", "b", "j"}
QUALIFIER_PATTERN = re.compile(
    r"^(?:.*风险|.*等级|[A-Za-z一二三四五六七八九十高中低重特]+级)$"
)


def load_stage4_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "stage4_atomic_modifier_relations",
        STAGE4_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载第四阶段代码：{STAGE4_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在第四阶段去维度原子ATT基础上，用确定性规则生成合并候选；"
            "不调用LLM，不改写原句。"
        )
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dimension-input",
        type=Path,
        default=DEFAULT_DIMENSION_INPUT,
    )
    parser.add_argument("--model", default="LTP/base2")
    parser.add_argument(
        "--segmentation-dictionary",
        type=Path,
        default=DEFAULT_DICTIONARY,
    )
    parser.add_argument("--segmentation-word-frequency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于0")
    return args


def overlaps_positions(
    start: int,
    end: int,
    excluded_positions: set[int],
) -> bool:
    return any(start <= position < end for position in excluded_positions)


def slice_tokens(
    sentence: str,
    spans: list[tuple[int, int]],
    start_index: int,
    end_index: int,
) -> str:
    start = spans[start_index][0]
    end = spans[end_index][1]
    return sentence[start:end].strip()


def relation_key(item: dict[str, object]) -> tuple[int, int]:
    return int(item["modifier_index"]), int(item["head_index"])


def build_graph(
    relations: list[dict[str, object]],
) -> tuple[dict[int, set[int]], set[int]]:
    incoming: dict[int, set[int]] = {}
    modifiers: set[int] = set()
    for relation in relations:
        modifier, head = relation_key(relation)
        incoming.setdefault(head, set()).add(modifier)
        modifiers.add(modifier)
    return incoming, modifiers


def collect_ancestors(
    head: int,
    incoming: dict[int, set[int]],
) -> set[int]:
    ancestors: set[int] = set()
    stack = list(incoming.get(head, ()))
    while stack:
        node = stack.pop()
        if node in ancestors or node == head:
            continue
        ancestors.add(node)
        stack.extend(incoming.get(node, ()))
    return ancestors


def is_safe_interval(
    words: list[str],
    pos_tags: list[str],
    selected: set[int],
    start: int,
    end: int,
) -> bool:
    for index in range(start, end + 1):
        if index in selected:
            continue
        word = words[index]
        if word in BRIDGE_WORDS:
            continue
        if word in QUERY_WORDS:
            return False
        if pos_tags[index] == "wp":
            return False
        return False
    return True


def add_candidate(
    candidates: list[dict[str, object]],
    seen: set[tuple[str, str]],
    *,
    modifier: str,
    head: str,
    source: str,
    confidence: str,
    evidence: str,
    head_index: int,
) -> None:
    modifier = modifier.strip()
    head = head.strip()
    if not modifier or not head or modifier == head:
        return
    key = (modifier, head)
    if key in seen:
        return
    seen.add(key)
    candidates.append(
        {
            "modifier": modifier,
            "head": head,
            "source": source,
            "confidence": confidence,
            "evidence": evidence,
            "head_index": head_index,
        }
    )


def graph_merge_candidates(
    sentence: str,
    words: list[str],
    pos_tags: list[str],
    spans: list[tuple[int, int]],
    relations: list[dict[str, object]],
    excluded_positions: set[int],
) -> list[dict[str, object]]:
    incoming, modifier_nodes = build_graph(relations)
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for head in sorted(incoming):
        if head in modifier_nodes:
            continue
        ancestors = {
            index
            for index in collect_ancestors(head, incoming)
            if index < head
        }
        participating_edges = [
            relation
            for relation in relations
            if int(relation["modifier_index"]) in ancestors
            and (
                int(relation["head_index"]) in ancestors
                or int(relation["head_index"]) == head
            )
        ]
        if len(participating_edges) < 2 or not ancestors:
            continue
        start = min(ancestors)
        if not is_safe_interval(
            words,
            pos_tags,
            ancestors,
            start,
            head - 1,
        ):
            continue
        modifier_start = spans[start][0]
        modifier_end = spans[head][0]
        head_start, head_end = spans[head]
        if overlaps_positions(
            modifier_start,
            modifier_end,
            excluded_positions,
        ) or overlaps_positions(head_start, head_end, excluded_positions):
            continue
        modifier = sentence[modifier_start:modifier_end].strip()
        add_candidate(
            candidates,
            seen,
            modifier=modifier,
            head=words[head],
            source="atomic_graph",
            confidence="medium",
            evidence=f"{len(participating_edges)}条连通原子ATT",
            head_index=head,
        )
    return candidates


def srl_merge_candidates(
    sentence: str,
    words: list[str],
    pos_tags: list[str],
    spans: list[tuple[int, int]],
    relations: list[dict[str, object]],
    srl_candidates: list[dict[str, object]],
    excluded_positions: set[int],
) -> list[dict[str, object]]:
    incoming, _ = build_graph(relations)
    outgoing: dict[int, set[int]] = {}
    for relation in relations:
        modifier, head = relation_key(relation)
        outgoing.setdefault(modifier, set()).add(head)
    relation_keys = {relation_key(relation) for relation in relations}
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in srl_candidates:
        recovered = str(item.get("recovered_modifier", "")).strip()
        if not recovered or not recovered.endswith("的"):
            continue
        predicate_index = int(item["modifier_index"])
        head_index = int(item["head_index"])
        if (predicate_index, head_index) not in relation_keys:
            continue
        modifier_start = sentence.find(recovered)
        if modifier_start < 0:
            continue
        modifier_end = modifier_start + len(recovered)
        modifier_token_indexes = [
            index
            for index, (start, end) in enumerate(spans)
            if start < modifier_end and modifier_start < end
        ]
        if any(words[index] in QUERY_WORDS for index in modifier_token_indexes):
            continue
        # SRL偶尔把已删除的地域/组织前缀一并收入A0。只有当维度重叠严格
        # 位于片段开头时才裁掉前缀；中间或尾部重叠仍拒绝，避免拼接改写。
        retained_indexes = [
            index
            for index in modifier_token_indexes
            if not overlaps_positions(
                spans[index][0],
                spans[index][1],
                excluded_positions,
            )
        ]
        if not retained_indexes:
            continue
        trimmed_start = spans[min(retained_indexes)][0]
        if any(
            overlaps_positions(
                spans[index][0],
                spans[index][1],
                excluded_positions,
            )
            for index in modifier_token_indexes
            if spans[index][0] >= trimmed_start
        ):
            continue
        recovered = sentence[trimmed_start:modifier_end].strip()
        modifier_start = trimmed_start
        if (
            not recovered.endswith("的")
            or recovered in GENERIC_VERBAL_MODIFIERS
        ):
            continue

        # 若SRL的原中心仍是更完整实体的内部成分，沿唯一原子边向右提升；
        # 数量/比率等指标不属于实体名称，因此不提升到指标中心。
        promoted_head = head_index
        visited = {promoted_head}
        while True:
            next_heads = {
                index
                for index in outgoing.get(promoted_head, set())
                if index > promoted_head
                and QUALIFIER_PATTERN.fullmatch(words[index]) is None
                and not words[index].endswith(("数", "率", "量"))
            }
            if len(next_heads) != 1:
                break
            next_head = next(iter(next_heads))
            if next_head in visited:
                break
            visited.add(next_head)
            promoted_head = next_head

        head_components = {
            index
            for index in collect_ancestors(promoted_head, incoming)
            if modifier_end <= spans[index][0] < spans[promoted_head][0]
        }
        selected = head_components | {promoted_head}
        if not selected:
            continue
        start_index = min(selected)
        if spans[start_index][0] < modifier_end:
            continue
        if not is_safe_interval(
            words,
            pos_tags,
            selected,
            start_index,
            promoted_head,
        ):
            continue
        head_start = spans[start_index][0]
        head_end = spans[promoted_head][1]
        if overlaps_positions(head_start, head_end, excluded_positions):
            continue
        add_candidate(
            candidates,
            seen,
            modifier=recovered,
            head=sentence[head_start:head_end],
            source="srl_explicit_de",
            confidence="high",
            evidence="第四阶段已接受SRL连续片段，并扩展连续实体中心",
            head_index=promoted_head,
        )
    return candidates


def is_qualifier(word: str, pos: str) -> bool:
    return (
        pos in ATTRIBUTE_POS
        or QUALIFIER_PATTERN.fullmatch(word) is not None
    )


def compact_entity_candidates(
    sentence: str,
    words: list[str],
    pos_tags: list[str],
    spans: list[tuple[int, int]],
    relations: list[dict[str, object]],
    excluded_positions: set[int],
) -> list[dict[str, object]]:
    incoming, modifier_nodes = build_graph(relations)
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for head, direct_modifiers in sorted(incoming.items()):
        if head in modifier_nodes:
            continue
        left_modifiers = sorted(
            index for index in direct_modifiers if index < head
        )
        qualifiers = [
            index
            for index in left_modifiers
            if is_qualifier(words[index], pos_tags[index])
        ]
        components = [
            index
            for index in left_modifiers
            if index not in qualifiers
        ]
        if not qualifiers or not components:
            continue

        # “的”左边通常是另一层作用域，不并入紧凑实体中心。
        last_de = max(
            (
                index
                for index in range(min(left_modifiers), head)
                if words[index] == "的"
            ),
            default=-1,
        )
        components = [index for index in components if index > last_de]
        if not components:
            continue
        head_start_index = min(components)
        head_selected = set(components) | set(qualifiers) | {head}
        if not is_safe_interval(
            words,
            pos_tags,
            head_selected,
            head_start_index,
            head,
        ):
            continue
        head_start = spans[head_start_index][0]
        head_end = spans[head][1]
        if overlaps_positions(head_start, head_end, excluded_positions):
            continue
        expanded_head = sentence[head_start:head_end]
        for qualifier in qualifiers:
            qualifier_start, qualifier_end = spans[qualifier]
            if overlaps_positions(
                qualifier_start,
                qualifier_end,
                excluded_positions,
            ):
                continue
            # 中心词扩展不能把该属性词本身再次包含进去。
            reduced_components = [
                index for index in components if index > qualifier
            ]
            if not reduced_components:
                continue
            reduced_start = min(reduced_components)
            selected = set(reduced_components) | {head}
            if not is_safe_interval(
                words,
                pos_tags,
                selected,
                reduced_start,
                head,
            ):
                continue
            expanded_head = slice_tokens(
                sentence,
                spans,
                reduced_start,
                head,
            )
            add_candidate(
                candidates,
                seen,
                modifier=words[qualifier],
                head=expanded_head,
                source="compact_entity",
                confidence="medium",
                evidence="属性型原子ATT + 连续紧凑实体成分",
                head_index=head,
            )
    return candidates


def select_merged_results(
    graph: list[dict[str, object]],
    srl_merged: list[dict[str, object]],
    compact: list[dict[str, object]],
    metric_head_pattern: re.Pattern[str],
) -> list[dict[str, object]]:
    """选择可自动接受的规则合并；原子图的非指标中心仅保留作观察。"""
    selected: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    higher_priority_heads = {
        int(item["head_index"]) for item in [*srl_merged, *compact]
    }
    for candidate in [*srl_merged, *compact]:
        key = (str(candidate["modifier"]), str(candidate["head"]))
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
    for candidate in graph:
        key = (str(candidate["modifier"]), str(candidate["head"]))
        if key in seen:
            continue
        if int(candidate["head_index"]) in higher_priority_heads:
            continue
        if "的" in str(candidate["modifier"]):
            continue
        if metric_head_pattern.fullmatch(str(candidate["head"])) is None:
            continue
        seen.add(key)
        selected.append(candidate)
    return selected


def format_plain_relations(items: list[dict[str, object]]) -> str:
    return "；".join(
        f"{item['modifier']} → {item['head']}" for item in items
    ) or "无"


def format_candidates(
    items: list[dict[str, object]],
    *,
    with_reason: bool = False,
) -> str:
    if not items:
        return "无"
    if not with_reason:
        return format_plain_relations(items)
    return "；".join(
        f"{item['modifier']} → {item['head']}"
        f"（{item['source']}/{item['confidence']}：{item['evidence']}）"
        for item in items
    )


def format_merged_or_atomic(
    merged: list[dict[str, object]],
    atomic: list[dict[str, object]],
) -> str:
    """有合并则展示合并结果，否则原样回填去维度原子ATT。"""
    if merged:
        return format_plain_relations(merged)
    return format_plain_relations(atomic)


def escape_table_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    args = parse_args()
    stage4 = load_stage4_module()
    segmentation_words = stage4.load_segmentation_words(
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

    output_rows: list[dict[str, object]] = []
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
            analysis = stage4.analyze_sentence(
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
            alignment = stage4.align_excluded_dimension_positions(
                sentence,
                dimension_sentence,
                missing_as_empty=dimension_missing,
            )
            filtered_relations, _ = (
                stage4.filter_atomic_relations_by_dimensions(
                    analysis["repaired_atomic_att"],
                    analysis["token_spans"],
                    alignment["excluded_positions"],
                )
            )
            clean_words = [stage4.clean_token(word) for word in words]
            clean_pos = [stage4.clean_token(pos) for pos in pos_tags]
            graph = graph_merge_candidates(
                sentence,
                clean_words,
                clean_pos,
                analysis["token_spans"],
                filtered_relations,
                alignment["excluded_positions"],
            )
            srl_merged = srl_merge_candidates(
                sentence,
                clean_words,
                clean_pos,
                analysis["token_spans"],
                filtered_relations,
                analysis["candidates"],
                alignment["excluded_positions"],
            )
            compact = compact_entity_candidates(
                sentence,
                clean_words,
                clean_pos,
                analysis["token_spans"],
                filtered_relations,
                alignment["excluded_positions"],
            )
            merged = select_merged_results(
                graph,
                srl_merged,
                compact,
                stage4.METRIC_HEAD_PATTERN,
            )
            output_rows.append(
                {
                    "source_line": line_number,
                    "sentence": sentence,
                    "atomic": filtered_relations,
                    "graph": graph,
                    "srl": srl_merged,
                    "compact": compact,
                    "merged": merged,
                }
            )
        print(f"已完成 {min(start + len(batch), len(records))}/{len(records)}")

    graph_count = sum(len(row["graph"]) for row in output_rows)
    srl_count = sum(len(row["srl"]) for row in output_rows)
    compact_count = sum(len(row["compact"]) for row in output_rows)
    merged_count = sum(len(row["merged"]) for row in output_rows)
    rows_with_merge = sum(bool(row["merged"]) for row in output_rows)

    lines = [
        "# 第4.1阶段：无LLM原子ATT规则合并实验",
        "",
        f"- 来源：`{stage4.display_project_path(args.input)}`",
        "- 输入关系：第四阶段去除维度原子ATT。",
        f"- 模型：`{args.model}`（只复现第四阶段分析，不调用LLM）。",
        "- 约束：结果必须来自原句连续片段；不补词、不调序、"
        "不重新引入已删除维度。",
        "- 定位：中间三路用于审计；最后一列只自动接受SRL完整动词定语、"
        "属性—实体合并，以及中心词为指标的原子图合并。",
        "- 回填规则：未触发合并时，最后一列原样保留"
        "“第四阶段去除维度原子ATT”。",
        f"- 统计：{len(output_rows)}句，{rows_with_merge}句产生合并候选；"
        f"原子图{graph_count}条，SRL合并{srl_count}条，"
        f"属性-实体合并{compact_count}条，去重后{merged_count}条。",
        "",
        "| 原文件行号 | 原句 | 第四阶段去除维度原子ATT | "
        "原子图连续合并 | SRL动词定语合并 | 属性-实体合并 | "
        "规则合并结果 |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['source_line']} | "
            f"{escape_table_cell(row['sentence'])} | "
            f"{escape_table_cell(format_plain_relations(row['atomic']))} | "
            f"{escape_table_cell(format_candidates(row['graph']))} | "
            f"{escape_table_cell(format_candidates(row['srl']))} | "
            f"{escape_table_cell(format_candidates(row['compact']))} | "
            f"{escape_table_cell(format_merged_or_atomic(row['merged'], row['atomic']))} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"已写入 {args.output}：{rows_with_merge}句产生"
        f"{merged_count}条规则合并结果"
    )


if __name__ == "__main__":
    main()
