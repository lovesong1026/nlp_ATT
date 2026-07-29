from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("merge_atomic_modifier_relations.py")
SPEC = importlib.util.spec_from_file_location("merge_rules", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载测试目标：{SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeRulesTest(unittest.TestCase):
    def test_graph_merges_chain_and_parallel_edges(self) -> None:
        sentence = "高风险变更操作"
        words = ["高风险", "变更", "操作"]
        spans = [(0, 3), (3, 5), (5, 7)]
        relations = [
            {"modifier_index": 0, "head_index": 2},
            {"modifier_index": 1, "head_index": 2},
        ]
        result = MODULE.graph_merge_candidates(
            sentence,
            words,
            ["a", "v", "v"],
            spans,
            relations,
            set(),
        )
        self.assertEqual(
            MODULE.format_candidates(result),
            "高风险变更 → 操作",
        )

    def test_compact_entity_expands_head(self) -> None:
        sentence = "高风险变更操作"
        words = ["高风险", "变更", "操作"]
        spans = [(0, 3), (3, 5), (5, 7)]
        relations = [
            {"modifier_index": 0, "head_index": 2},
            {"modifier_index": 1, "head_index": 2},
        ]
        result = MODULE.compact_entity_candidates(
            sentence,
            words,
            ["a", "v", "v"],
            spans,
            relations,
            set(),
        )
        self.assertEqual(
            MODULE.format_candidates(result),
            "高风险 → 变更操作",
        )

    def test_excluded_dimension_blocks_merge(self) -> None:
        result = MODULE.graph_merge_candidates(
            "全球运营商业务",
            ["全球", "运营商", "业务"],
            ["n", "n", "n"],
            [(0, 2), (2, 5), (5, 7)],
            [
                {"modifier_index": 0, "head_index": 1},
                {"modifier_index": 1, "head_index": 2},
            ],
            {0},
        )
        self.assertEqual(result, [])

    def test_only_metric_graph_merge_is_selected(self) -> None:
        graph = [
            {
                "modifier": "服务收入",
                "head": "完成率",
                "head_index": 2,
            },
            {
                "modifier": "交付EI",
                "head": "项目",
                "head_index": 5,
            },
            {
                "modifier": "导致的事故",
                "head": "数",
                "head_index": 8,
            },
        ]
        result = MODULE.select_merged_results(
            graph,
            [],
            [],
            re.compile(r"^(?:.*率|数量|次数|数)$"),
        )
        self.assertEqual(
            [(item["modifier"], item["head"]) for item in result],
            [("服务收入", "完成率")],
        )

    def test_srl_trims_dimension_prefix_and_promotes_entity_head(self) -> None:
        sentence = "全球供方问题导致的二级管理升级单"
        words = [
            "全球",
            "供方",
            "问题",
            "导致",
            "的",
            "二级",
            "管理",
            "升级单",
        ]
        spans = [
            (0, 2),
            (2, 4),
            (4, 6),
            (6, 8),
            (8, 9),
            (9, 11),
            (11, 13),
            (13, 16),
        ]
        relations = [
            {"modifier_index": 3, "head_index": 6},
            {"modifier_index": 5, "head_index": 6},
            {"modifier_index": 6, "head_index": 7},
        ]
        result = MODULE.srl_merge_candidates(
            sentence,
            words,
            ["n", "n", "n", "v", "u", "b", "v", "n"],
            spans,
            relations,
            [
                {
                    "modifier_index": 3,
                    "head_index": 6,
                    "recovered_modifier": "全球供方问题导致的",
                }
            ],
            {0, 1},
        )
        self.assertEqual(
            MODULE.format_candidates(result),
            "供方问题导致的 → 二级管理升级单",
        )

    def test_falls_back_to_atomic_when_nothing_is_merged(self) -> None:
        atomic = [{"modifier": "负", "head": "增长"}]
        self.assertEqual(
            MODULE.format_merged_or_atomic([], atomic),
            "负 → 增长",
        )

    def test_merged_output_does_not_include_audit_annotation(self) -> None:
        merged = [
            {
                "modifier": "服务收入",
                "head": "完成率",
                "source": "atomic_graph",
                "confidence": "medium",
                "evidence": "2条连通原子ATT",
            }
        ]
        self.assertEqual(
            MODULE.format_merged_or_atomic(merged, []),
            "服务收入 → 完成率",
        )

    def test_recursive_qualifiers_keep_independent_levels(self) -> None:
        result = MODULE.compact_entity_candidates(
            "S级风险交付项目",
            ["S级", "风险", "交付", "项目"],
            ["b", "n", "v", "n"],
            [(0, 2), (2, 4), (4, 6), (6, 8)],
            [
                {"modifier_index": 0, "head_index": 1},
                {"modifier_index": 1, "head_index": 3},
                {"modifier_index": 2, "head_index": 3},
            ],
            set(),
        )
        self.assertEqual(
            MODULE.format_candidates(result),
            "S级 → 风险交付项目；风险 → 交付项目",
        )

    def test_explicit_de_builds_lexical_modifier_block(self) -> None:
        result = MODULE.compact_entity_candidates(
            "低风险的网络质量比拼网络",
            ["低", "风险", "的", "网络", "质量", "比拼", "网络"],
            ["a", "n", "u", "n", "n", "v", "n"],
            [(0, 1), (1, 3), (3, 4), (4, 6), (6, 8), (8, 10), (10, 12)],
            [
                {"modifier_index": 0, "head_index": 1},
                {"modifier_index": 1, "head_index": 6},
                {"modifier_index": 3, "head_index": 4},
                {"modifier_index": 4, "head_index": 5},
                {"modifier_index": 5, "head_index": 6},
            ],
            set(),
        )
        self.assertEqual(
            MODULE.format_candidates(result),
            "低风险的 → 网络质量比拼网络",
        )

    def test_coordination_propagates_same_entity_head(self) -> None:
        result = MODULE.compact_entity_candidates(
            "中风险或高风险的项目",
            ["中风险", "或", "高风险", "的", "项目"],
            ["v", "c", "a", "u", "n"],
            [(0, 3), (3, 4), (4, 7), (7, 8), (8, 10)],
            [{"modifier_index": 0, "head_index": 4}],
            set(),
        )
        self.assertEqual(
            MODULE.format_candidates(result),
            "中风险 → 项目；高风险 → 项目",
        )

    def test_postposed_cause_recovers_original_contiguous_span(self) -> None:
        result = MODULE.structural_surface_candidates(
            "有多少个风险交付项目是由于物料供应问题导致的",
            ["有", "多少", "个", "风险", "交付", "项目", "是",
             "由于", "物料", "供应", "问题", "导致", "的"],
            ["v", "r", "q", "n", "v", "n", "v",
             "p", "n", "v", "n", "v", "u"],
            [(0, 1), (1, 3), (3, 4), (4, 6), (6, 8), (8, 10),
             (10, 11), (11, 13), (13, 15), (15, 17), (17, 19),
             (19, 21), (21, 22)],
            [
                {"modifier_index": 3, "head_index": 5},
                {"modifier_index": 4, "head_index": 5},
            ],
            set(),
        )
        self.assertEqual(
            MODULE.format_candidates(result),
            "由于物料供应问题导致的 → 风险交付项目",
        )

    def test_more_complete_modifier_dominates_same_head_suffix(self) -> None:
        selected = MODULE.select_merged_results(
            [],
            [{"modifier": "未定性的", "head": "二级事故", "head_index": 4}],
            [{"modifier": "根因未定性的", "head": "二级事故", "head_index": 4}],
            re.compile(r"率$"),
        )
        self.assertEqual(
            [(item["modifier"], item["head"]) for item in selected],
            [("根因未定性的", "二级事故")],
        )


if __name__ == "__main__":
    unittest.main()
