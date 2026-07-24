from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ltp import LTP


# 运行命令：
# python extract_attributives_ltp.py \
#   data/original_question.md \
#   output/original_question_attributives_ltp_reconstructed.md \
#   --model LTP/base \
#   --use-segmentation-words \
#   --reconstruct-modifiers \
#   --jsonl-output output/original_question_attributives_ltp_reconstructed.jsonl


# 从 data/original_question.md 提取的最小不可拆业务语义单元。
# 这里只保护分词，不放入“服务收入完成率”“网络变更操作成功率”等
# 仍需要分析内部结构的完整指标或长业务短语。
SEGMENTATION_WORDS: list[str] = [
    # 地域、组织和业务范围
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
    # 等级和风险；否定/状态短语留给后续完整定语重建
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
    # 原因、质量和资源类别
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
    # 固定业务事件、场景和能力单元
    "国家三领先",
    "领导力践行",
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
    # 指标中应保持完整的基本词
    "完成率",
    "成功率",
    "及时率",
    "成本率",
    "利润率",
    "销毛率",
    "成熟度",
    "占比",
    "根因",
    # 英文缩写、产品名和专名；长词放在共享前缀的短词前面
    "FaceboY",
    "SmartCare",
    "Facility",
    "Telkom",
    "TOP3",
    "EHS",
    "NPX",
    "AMS",
    "OYla",
    "MBB",
    "FBB",
    "ITS",
    "NIS",
    "SEC",
    "P3",
    "EI",
    "5G",
    "H1",
    "DC",
    "BG",
    "FB",
    "IT",
]

SCOPE_WORDS = {
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
CLAUSE_MARKERS = {"由于", "由"}
PUNCTUATION_PATTERN = re.compile(r"^[，,。！？!?；;：:、]$")
YEAR_TOKEN_PATTERN = re.compile(r"^(?:\d{4}|本|当|去|今|明)年$")
MONTH_TOKEN_PATTERN = re.compile(
    r"^(?:\d{1,2}|[一二三四五六七八九十]+)月$"
)
NOISE_TOKEN_PATTERN = re.compile(
    r"^(?:"
    r"这|这些|这个|该|此|上述|前述|"
    r"多少|哪个|哪些|哪|几|什么|"
    r"\d+|[一二两三四五六七八九十百两]+|"
    r"个|起|次|张|项|条|份"
    r")$"
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 LTP 提取逐句定中（ATT）关系。")
    parser.add_argument("input", type=Path, help="每行一句的 Markdown 文本")
    parser.add_argument("output", type=Path, help="Markdown 标注结果")
    parser.add_argument("--model", default="LTP/base", help="LTP 模型名称或路径")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--use-segmentation-words",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用内置 SEGMENTATION_WORDS 保护领域词分词（默认关闭）",
    )
    parser.add_argument(
        "--segmentation-word-frequency",
        type=int,
        default=2,
        help="注册领域词时使用的词频权重（默认 2）",
    )
    parser.add_argument(
        "--reconstruct-modifiers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="基于依存子树重建原句中的完整定语候选（默认关闭）",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        help="可选：保存分词、依存关系、原始 ATT 和完整定语候选的 JSONL 文件",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.segmentation_word_frequency < 1:
        parser.error("--segmentation-word-frequency 必须大于 0")
    return args


def escape_table_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def clean_token(word: str) -> str:
    return word.strip()


def locate_token_spans(sentence: str, words: list[str]) -> list[tuple[int, int]]:
    """按 LTP 分词顺序将 token 对齐回原句，区间为左闭右开。"""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for raw_word in words:
        word = clean_token(raw_word)
        if not word:
            raise ValueError(f"发现空 token：{raw_word!r}")
        start = sentence.find(word, cursor)
        if start < 0:
            raise ValueError(
                f"token 无法对齐回原句：token={word!r}, cursor={cursor}, "
                f"sentence={sentence!r}"
            )
        end = start + len(word)
        spans.append((start, end))
        cursor = end
    return spans


def build_children(heads: list[int]) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in heads]
    for child_index, head in enumerate(heads):
        if head > 0:
            children[head - 1].append(child_index)
    return children


def descendants(root: int, children: list[list[int]]) -> set[int]:
    result: set[int] = set()
    stack = list(children[root])
    while stack:
        index = stack.pop()
        if index in result:
            continue
        result.add(index)
        stack.extend(children[index])
    return result


def is_noise_token(word: str) -> bool:
    return (
        not word
        or word in SCOPE_WORDS
        or PUNCTUATION_PATTERN.fullmatch(word) is not None
        or NOISE_TOKEN_PATTERN.fullmatch(word) is not None
    )


def is_low_value_relation(modifier: str, head: str) -> bool:
    """过滤不适合业务 RAG 检索的数量指示和日历内部修饰关系。"""
    return (
        NOISE_TOKEN_PATTERN.fullmatch(modifier) is not None
        or (
            YEAR_TOKEN_PATTERN.fullmatch(modifier) is not None
            and MONTH_TOKEN_PATTERN.fullmatch(head) is not None
        )
    )


def collect_content_branch(
    root: int,
    modifier: int,
    words: list[str],
    labels: list[str],
    children: list[list[int]],
) -> set[int]:
    """收集定语核心词左侧的业务内容分支，并在范围词或数量噪声处剪枝。"""
    selected: set[int] = set()
    stack = [root]
    while stack:
        index = stack.pop()
        word = clean_token(words[index])
        if index > modifier or is_noise_token(word):
            continue
        selected.add(index)
        for child in children[index]:
            if child <= modifier and labels[child] in CONTENT_CHILD_LABELS:
                stack.append(child)
    return selected


def reconstruct_att_candidates(
    sentence: str,
    words: list[str],
    heads: list[int],
    labels: list[str],
    spans: list[tuple[int, int]],
) -> list[dict[str, object]]:
    children = build_children(heads)
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()

    for modifier, (head, label) in enumerate(zip(heads, labels, strict=True)):
        if label != "ATT" or head == 0:
            continue

        head_index = head - 1
        core_text = clean_token(words[modifier])
        head_text = clean_token(words[head_index])
        subtree = descendants(modifier, children)
        rad_children = sorted(
            index
            for index in children[modifier]
            if labels[index] == "RAD"
            and clean_token(words[index]) == "的"
            and index > modifier
        )

        selected = {modifier}
        source = "ltp_token"
        end_token = modifier

        if rad_children:
            source = "ltp_subtree"
            end_token = rad_children[0]
            markers = sorted(
                index
                for index in subtree
                if index < modifier and clean_token(words[index]) in CLAUSE_MARKERS
            )
            if markers:
                # 取离定语核心最近的“由于/由”，避免把更外层上下文带入。
                start_token = markers[-1]
                selected.update(
                    index
                    for index in subtree
                    if start_token <= index <= end_token
                )
            else:
                for child in children[modifier]:
                    if (
                        child < modifier
                        and labels[child] in CONTENT_CHILD_LABELS
                        and not is_noise_token(clean_token(words[child]))
                    ):
                        selected.update(
                            collect_content_branch(
                                child, modifier, words, labels, children
                            )
                        )
                start_token = min(selected)

            # 依存子树偶尔跨越标点；完整定语不能越过最近的句内边界。
            punctuation_before_core = [
                index
                for index in range(start_token, modifier)
                if PUNCTUATION_PATTERN.fullmatch(clean_token(words[index]))
            ]
            if punctuation_before_core:
                start_token = punctuation_before_core[-1] + 1
        else:
            start_token = modifier
            if is_low_value_relation(core_text, head_text):
                continue

        start_char = spans[start_token][0]
        end_char = spans[end_token][1]
        modifier_text = sentence[start_char:end_char].strip()
        if not modifier_text:
            continue

        dedupe_key = (modifier_text, head_index)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(
            {
                "modifier_text": modifier_text,
                "modifier_core": core_text,
                "head_text": head_text,
                "modifier_token_index": modifier + 1,
                "head_token_index": head_index + 1,
                "start": start_char,
                "end": end_char,
                "source": source,
            }
        )

    # “项目是由于……导致的”属于后置条件，不会被 LTP 标成 ATT。
    # 这里通过“导致 → 是”和“项目 ↔ 是”的依存结构补为完整候选，
    # 文本仍只截取原句，不进行改写。
    for core, (head, label) in enumerate(zip(heads, labels, strict=True)):
        if label == "ATT" or head == 0:
            continue
        rad_children = sorted(
            index
            for index in children[core]
            if labels[index] == "RAD"
            and clean_token(words[index]) == "的"
            and index > core
        )
        if not rad_children:
            continue
        subtree = descendants(core, children)
        markers = sorted(
            index
            for index in subtree
            if index < core and clean_token(words[index]) in CLAUSE_MARKERS
        )
        copula = head - 1
        if not markers or clean_token(words[copula]) != "是":
            continue

        target_options = [
            index
            for index in children[copula]
            if index < copula and labels[index] in {"SBV", "DBL", "FOB", "VOB"}
        ]
        if not target_options and heads[copula] > 0:
            target_options = [
                index
                for index, sibling_head in enumerate(heads)
                if index < copula
                and sibling_head == heads[copula]
                and labels[index] in {"SBV", "DBL", "FOB", "VOB"}
            ]
        if not target_options:
            continue

        target = max(target_options)
        start_token = markers[-1]
        end_token = rad_children[0]
        start_char = spans[start_token][0]
        end_char = spans[end_token][1]
        modifier_text = sentence[start_char:end_char].strip()
        dedupe_key = (modifier_text, target)
        if not modifier_text or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(
            {
                "modifier_text": modifier_text,
                "modifier_core": clean_token(words[core]),
                "head_text": clean_token(words[target]),
                "modifier_token_index": core + 1,
                "head_token_index": target + 1,
                "start": start_char,
                "end": end_char,
                "source": "ltp_copular_clause",
            }
        )

    candidates.sort(
        key=lambda item: (
            int(item["head_token_index"]),
            int(item["start"]),
            int(item["end"]),
        )
    )
    return candidates


def main() -> None:
    args = parse_args()
    source_lines = args.input.read_text(encoding="utf-8").splitlines()

    records = [
        (line_number, text.strip())
        for line_number, text in enumerate(source_lines, start=1)
        if line_number > 1 and text.strip()
    ]

    model = LTP(args.model)
    if args.use_segmentation_words:
        model.add_words(
            SEGMENTATION_WORDS,
            freq=args.segmentation_word_frequency,
        )

    analyses: list[dict[str, object]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        result = model.pipeline([text for _, text in batch], tasks=["cws", "dep"])

        for (line_number, sentence), words, dependency in zip(
            batch, result.cws, result.dep, strict=True
        ):
            heads = [int(head) for head in dependency["head"]]
            labels = [str(label) for label in dependency["label"]]
            spans = locate_token_spans(sentence, words)
            tokens = [
                {
                    "index": index + 1,
                    "text": clean_token(word),
                    "start": token_span[0],
                    "end": token_span[1],
                    "head": heads[index],
                    "label": labels[index],
                }
                for index, (word, token_span) in enumerate(
                    zip(words, spans, strict=True)
                )
            ]

            raw_att: list[dict[str, object]] = []
            for index, (head, label) in enumerate(
                zip(heads, labels, strict=True)
            ):
                if label != "ATT" or head == 0:
                    continue
                raw_att.append(
                    {
                        "modifier_text": clean_token(words[index]),
                        "head_text": clean_token(words[head - 1]),
                        "modifier_token_index": index + 1,
                        "head_token_index": head,
                        "start": spans[index][0],
                        "end": spans[index][1],
                    }
                )

            complete_candidates = (
                reconstruct_att_candidates(
                    sentence, words, heads, labels, spans
                )
                if args.reconstruct_modifiers
                else []
            )
            analyses.append(
                {
                    "source_line": line_number,
                    "sentence": sentence,
                    "tokens": tokens,
                    "raw_att": raw_att,
                    "complete_modifier_candidates": complete_candidates,
                }
            )

    if len(analyses) != len(records):
        raise RuntimeError(f"记录数不一致：输入 {len(records)}，输出 {len(analyses)}")

    output_lines = [
        f"# {args.input.stem} 定语提取结果",
        "",
        f"- 来源：`{args.input.as_posix()}`",
        f"- 模型：`{args.model}`",
        *(
            [
                "- 自定义词典："
                f"`SEGMENTATION_WORDS`（{len(SEGMENTATION_WORDS)} 词，"
                f"freq={args.segmentation_word_frequency}）。"
            ]
            if args.use_segmentation_words
            else ["- 自定义词典：未使用。"]
        ),
        "- 规则：提取 LTP 依存句法标签 `ATT`，格式为“修饰语 → 中心词”。",
        *(
            [
                "- 完整定语候选：从原句字符区间截取，不做语义改写；"
                "过滤数量指示噪声，并在范围词处停止扩展。"
            ]
            if args.reconstruct_modifiers
            else []
        ),
        "- 说明：结果为模型自动标注；领域短语和歧义句建议结合业务语义人工复核。",
        "",
        *(
            [
                "| 原文件行号 | 原句 | LTP 原始 ATT | 完整定语候选 |",
                "|---:|---|---|---|",
            ]
            if args.reconstruct_modifiers
            else [
                "| 原文件行号 | 原句 | 定语（修饰语 → 中心词） |",
                "|---:|---|---|",
            ]
        ),
    ]

    for analysis in analyses:
        raw_annotation = "；".join(
            f"{item['modifier_text']} → {item['head_text']}"
            for item in analysis["raw_att"]
        ) or "无"
        if args.reconstruct_modifiers:
            complete_annotation = "；".join(
                f"{item['modifier_text']} → {item['head_text']}"
                for item in analysis["complete_modifier_candidates"]
            ) or "无"
            output_lines.append(
                f"| {analysis['source_line']} | "
                f"{escape_table_cell(str(analysis['sentence']))} | "
                f"{escape_table_cell(raw_annotation)} | "
                f"{escape_table_cell(complete_annotation)} |"
            )
        else:
            output_lines.append(
                f"| {analysis['source_line']} | "
                f"{escape_table_cell(str(analysis['sentence']))} | "
                f"{escape_table_cell(raw_annotation)} |"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    if args.jsonl_output:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_output.write_text(
            "".join(
                json.dumps(analysis, ensure_ascii=False) + "\n"
                for analysis in analyses
            ),
            encoding="utf-8",
        )
        print(f"已写入 {args.jsonl_output}：{len(analyses)} 句")
    if args.use_segmentation_words:
        print(
            f"已加载 SEGMENTATION_WORDS：{len(SEGMENTATION_WORDS)} 词，"
            f"freq={args.segmentation_word_frequency}"
        )
    print(f"已写入 {args.output}：{len(records)} 句")


if __name__ == "__main__":
    main()
