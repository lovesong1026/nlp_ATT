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


if __name__ == "__main__":
    unittest.main()
