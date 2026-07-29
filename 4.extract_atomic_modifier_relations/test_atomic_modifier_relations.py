from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("extract_atomic_modifier_relations.py")
SPEC = importlib.util.spec_from_file_location(
    "stage4_atomic_modifier_relations",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载模块：{MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def direct_item(
    modifier_index: int,
    head_index: int,
    words: list[str],
    pos: list[str],
) -> dict[str, object]:
    return {
        "modifier_index": modifier_index,
        "modifier": words[modifier_index],
        "modifier_pos": pos[modifier_index],
        "head_index": head_index,
        "head": words[head_index],
        "head_pos": pos[head_index],
    }


def relations(
    words: list[str],
    pos: list[str],
    heads: list[int],
    labels: list[str],
    direct: list[dict[str, object]],
    srl: dict[int, dict[str, object]] | None = None,
    sdp_heads: list[int] | None = None,
    sdp_labels: list[str] | None = None,
) -> set[tuple[str, str]]:
    _, _, final = MODULE.build_atomic_att_repairs(
        words,
        pos,
        heads,
        labels,
        direct,
        srl or {},
        {"高风险", "中风险"},
        sdp_heads,
        sdp_labels,
    )
    return {
        (str(item["modifier"]), str(item["head"]))
        for item in final
    }


class AtomicAttRepairTests(unittest.TestCase):
    def test_adjacent_lexical_modifier_recovers_negative_growth(self) -> None:
        words = ["收入", "同比", "负", "增长"]
        pos = ["n", "j", "b", "v"]
        heads = [2, 4, 4, 0]
        labels = ["ATT", "ADV", "ADV", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(0, 1, words, pos)],
            sdp_heads=[4, 4, 4, 0],
            sdp_labels=["EXP", "MANN", "MANN", "Root"],
        )
        self.assertIn(("负", "增长"), result)

    def test_semantic_polarity_keeps_negation(self) -> None:
        words = ["未", "关闭", "项目"]
        pos = ["d", "v", "n"]
        heads = [2, 3, 0]
        labels = ["ADV", "ATT", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(1, 2, words, pos)],
            sdp_heads=[2, 3, 0],
            sdp_labels=["mNEG", "rEXP", "Root"],
        )
        self.assertIn(("未", "关闭"), result)

    def test_sdp_feature_recovers_metric_modifier(self) -> None:
        words = ["收入", "预算", "完成率"]
        pos = ["n", "n", "n"]
        heads = [3, 3, 0]
        labels = ["FOB", "FOB", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [],
            sdp_heads=[3, 3, 0],
            sdp_labels=["FEAT", "FEAT", "Root"],
        )
        self.assertIn(("收入", "完成率"), result)
        self.assertIn(("预算", "完成率"), result)

    def test_fob_modifier_lifts_through_att_verb(self) -> None:
        words = ["风险", "交付", "项目"]
        pos = ["n", "v", "n"]
        heads = [2, 3, 0]
        labels = ["FOB", "ATT", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(1, 2, words, pos)],
        )
        self.assertIn(("风险", "项目"), result)

    def test_alphanumeric_compound_repairs_reversed_parse(self) -> None:
        words = ["FaceboY", "比拼", "网络"]
        pos = ["n", "v", "n"]
        heads = [2, 0, 2]
        labels = ["SBV", "HED", "VOB"]
        result = relations(words, pos, heads, labels, [])
        self.assertIn(("FaceboY", "比拼"), result)
        self.assertIn(("比拼", "网络"), result)

    def test_alphanumeric_object_lifts_to_nominal_head(self) -> None:
        words = ["交付", "EI", "项目"]
        pos = ["v", "ws", "n"]
        heads = [3, 1, 0]
        labels = ["ATT", "VOB", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(0, 2, words, pos)],
        )
        self.assertIn(("EI", "项目"), result)

    def test_position_word_cannot_be_content_head(self) -> None:
        self.assertFalse(
            MODULE.is_content_head("中", "nd", {"高风险", "中风险"})
        )

    def test_negated_predicate_reheads_to_inner_object(self) -> None:
        words = ["不", "满", "客户", "声音"]
        pos = ["a", "v", "n", "n"]
        heads = [2, 4, 2, 0]
        labels = ["ADV", "ATT", "VOB", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(1, 3, words, pos)],
            sdp_heads=[2, 3, 2, 0],
            sdp_labels=["mNEG", "rEXP", "PAT", "Root"],
        )
        self.assertNotIn(("满", "声音"), result)
        self.assertIn(("不", "满"), result)
        self.assertIn(("满", "客户"), result)
        self.assertIn(("客户", "声音"), result)

    def test_compact_np_recovers_high_risk(self) -> None:
        words = ["高风险", "交付", "项目"]
        pos = ["a", "v", "n"]
        heads = [2, 3, 0]
        labels = ["ADV", "ATT", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(1, 2, words, pos)],
        )
        self.assertEqual(
            result,
            {("高风险", "项目"), ("交付", "项目")},
        )

    def test_quantity_head_lifts_content_modifier(self) -> None:
        words = ["中风险", "个", "项目"]
        pos = ["nd", "q", "n"]
        heads = [2, 3, 0]
        labels = ["ATT", "ATT", "HED"]
        result = relations(words, pos, heads, labels, [])
        self.assertEqual(result, {("中风险", "项目")})

    def test_coordination_inherits_head(self) -> None:
        words = ["一级", "二级", "事故"]
        pos = ["b", "b", "n"]
        heads = [3, 1, 0]
        labels = ["ATT", "COO", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(0, 2, words, pos)],
        )
        self.assertEqual(
            result,
            {("一级", "事故"), ("二级", "事故")},
        )

    def test_backward_att_is_replaced(self) -> None:
        words = ["地区部", "的", "高风险", "交付", "项目"]
        pos = ["n", "u", "a", "v", "n"]
        heads = [5, 1, 1, 5, 0]
        labels = ["ATT", "RAD", "ATT", "ATT", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [
                direct_item(0, 4, words, pos),
                direct_item(2, 0, words, pos),
                direct_item(3, 4, words, pos),
            ],
        )
        self.assertNotIn(("高风险", "地区部"), result)
        self.assertIn(("高风险", "项目"), result)

    def test_srl_target_lifts_short_head(self) -> None:
        words = ["未", "关闭", "的", "管理", "升级单"]
        pos = ["d", "v", "u", "v", "n"]
        heads = [2, 4, 2, 5, 0]
        labels = ["ADV", "ATT", "RAD", "ATT", "HED"]
        srl = {
            1: {
                "predicate": "关闭",
                "index": 1,
                "arguments": [
                    {"role": "ARGM-ADV", "text": "未", "start": 0, "end": 0},
                    {
                        "role": "A1",
                        "text": "管理升级单",
                        "start": 3,
                        "end": 4,
                    },
                ],
            }
        }
        result = relations(
            words,
            pos,
            heads,
            labels,
            [
                direct_item(1, 3, words, pos),
                direct_item(3, 4, words, pos),
            ],
            srl,
        )
        self.assertNotIn(("关闭", "管理"), result)
        self.assertIn(("关闭", "升级单"), result)
        self.assertIn(("管理", "升级单"), result)

    def test_srl_target_lifts_short_head_without_dep_att_chain(self) -> None:
        words = ["未", "关闭", "的", "管理", "升级单"]
        pos = ["d", "v", "u", "v", "n"]
        heads = [2, 4, 2, 5, 0]
        labels = ["ADV", "ATT", "RAD", "FOB", "HED"]
        srl = {
            1: {
                "predicate": "关闭",
                "index": 1,
                "arguments": [
                    {"role": "ARGM-ADV", "text": "未", "start": 0, "end": 0},
                    {
                        "role": "A1",
                        "text": "管理升级单",
                        "start": 3,
                        "end": 4,
                    },
                ],
            }
        }
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(1, 3, words, pos)],
            srl,
            sdp_heads=[2, 4, 2, 5, 0],
            sdp_labels=["mNEG", "rPAT", "mDEPD", "FEAT", "Root"],
        )
        self.assertNotIn(("关闭", "管理"), result)
        self.assertIn(("关闭", "升级单"), result)
        self.assertIn(("管理", "升级单"), result)

    def test_srl_target_moves_from_metric_to_inner_event(self) -> None:
        words = ["未", "关闭", "的", "管理", "升级", "数量"]
        pos = ["d", "v", "u", "v", "v", "n"]
        heads = [2, 6, 2, 6, 6, 0]
        labels = ["ADV", "ATT", "RAD", "ATT", "ATT", "HED"]
        srl = {
            1: {
                "predicate": "关闭",
                "index": 1,
                "arguments": [
                    {
                        "role": "A1",
                        "text": "管理升级数量",
                        "start": 3,
                        "end": 5,
                    },
                ],
            }
        }
        result = relations(
            words,
            pos,
            heads,
            labels,
            [
                direct_item(1, 5, words, pos),
                direct_item(3, 5, words, pos),
                direct_item(4, 5, words, pos),
            ],
            srl,
        )
        self.assertNotIn(("关闭", "数量"), result)
        self.assertIn(("关闭", "升级"), result)

    def test_unmarked_coordination_recovers_compound_chain(self) -> None:
        words = ["二级", "管理", "升级", "数量"]
        pos = ["b", "v", "v", "n"]
        heads = [2, 4, 2, 0]
        labels = ["ATT", "ATT", "COO", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [
                direct_item(0, 1, words, pos),
                direct_item(1, 3, words, pos),
            ],
            sdp_heads=[3, 3, 4, 0],
            sdp_labels=["FEAT", "eCOO", "FEAT", "Root"],
        )
        self.assertIn(("管理", "升级"), result)
        self.assertIn(("升级", "数量"), result)
        self.assertNotIn(("管理", "数量"), result)

    def test_srl_argument_recovers_parallel_compound_members(self) -> None:
        words = ["区域", "运营商", "服务", "订货", "完成"]
        pos = ["n", "n", "v", "v", "v"]
        heads = [2, 5, 5, 3, 0]
        labels = ["ATT", "SBV", "FOB", "COO", "HED"]
        srl = {
            4: {
                "predicate": "完成",
                "index": 4,
                "arguments": [
                    {
                        "role": "A1",
                        "text": "区域运营商服务订货",
                        "start": 0,
                        "end": 3,
                    },
                ],
            }
        }
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(0, 1, words, pos)],
            srl,
        )
        self.assertIn(("运营商", "服务"), result)
        self.assertIn(("服务", "订货"), result)

    def test_nominalized_predicate_does_not_cross_explicit_de(self) -> None:
        words = ["质量", "导致", "的", "事故"]
        pos = ["n", "v", "u", "n"]
        heads = [2, 4, 2, 0]
        labels = ["SBV", "ATT", "RAD", "HED"]
        srl = {
            1: {
                "predicate": "导致",
                "index": 1,
                "arguments": [
                    {"role": "A0", "text": "质量", "start": 0, "end": 0},
                    {"role": "A1", "text": "事故", "start": 3, "end": 3},
                ],
            }
        }
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(1, 3, words, pos)],
            srl,
        )
        self.assertNotIn(("质量", "导致"), result)

    def test_srl_argument_does_not_turn_generic_predicate_into_head(self) -> None:
        words = ["交付", "项目", "存在", "风险"]
        pos = ["v", "n", "v", "n"]
        heads = [2, 3, 0, 3]
        labels = ["ATT", "SBV", "HED", "VOB"]
        srl = {
            2: {
                "predicate": "存在",
                "index": 2,
                "arguments": [
                    {
                        "role": "A0",
                        "text": "交付项目",
                        "start": 0,
                        "end": 1,
                    },
                    {"role": "A1", "text": "风险", "start": 3, "end": 3},
                ],
            }
        }
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(0, 1, words, pos)],
            srl,
        )
        self.assertNotIn(("项目", "存在"), result)

    def test_local_attribute_prefers_adjacent_event_head(self) -> None:
        words = ["高风险", "变更", "操作"]
        pos = ["a", "v", "v"]
        heads = [3, 3, 0]
        labels = ["ATT", "ATT", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [
                direct_item(0, 2, words, pos),
                direct_item(1, 2, words, pos),
            ],
        )
        self.assertIn(("高风险", "变更"), result)
        self.assertIn(("变更", "操作"), result)
        self.assertNotIn(("高风险", "操作"), result)

    def test_attribute_keeps_nominal_entity_head(self) -> None:
        words = ["高风险", "交付", "项目"]
        pos = ["a", "v", "n"]
        heads = [3, 3, 0]
        labels = ["ATT", "ATT", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [
                direct_item(0, 2, words, pos),
                direct_item(1, 2, words, pos),
            ],
        )
        self.assertIn(("高风险", "项目"), result)
        self.assertIn(("交付", "项目"), result)
        self.assertNotIn(("高风险", "交付"), result)

    def test_explicit_de_modifier_reheads_from_metric_to_entity(self) -> None:
        words = ["场景", "的", "风险", "项目", "数"]
        pos = ["n", "u", "n", "n", "n"]
        heads = [5, 1, 4, 5, 0]
        labels = ["ATT", "RAD", "ATT", "ATT", "HED"]
        result = relations(
            words,
            pos,
            heads,
            labels,
            [
                direct_item(0, 4, words, pos),
                direct_item(2, 3, words, pos),
                direct_item(3, 4, words, pos),
            ],
        )
        self.assertIn(("场景", "项目"), result)
        self.assertNotIn(("场景", "数"), result)

    def test_srl_compact_verbal_object_recovers_entity_head(self) -> None:
        words = ["中风险", "交付", "EI", "项目", "有"]
        pos = ["nd", "v", "ws", "n", "v"]
        heads = [5, 5, 4, 2, 0]
        labels = ["DBL", "VOB", "ATT", "VOB", "HED"]
        srl = {
            4: {
                "predicate": "有",
                "index": 4,
                "arguments": [
                    {
                        "role": "A1",
                        "text": "中风险交付EI项目",
                        "start": 0,
                        "end": 3,
                    },
                ],
            }
        }
        result = relations(
            words,
            pos,
            heads,
            labels,
            [direct_item(2, 3, words, pos)],
            srl,
        )
        self.assertIn(("交付", "项目"), result)
        self.assertIn(("中风险", "项目"), result)
        self.assertNotIn(("中风险", "交付"), result)


if __name__ == "__main__":
    unittest.main()
