from __future__ import annotations

import argparse
from pathlib import Path

from ltp import LTP


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
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.segmentation_word_frequency < 1:
        parser.error("--segmentation-word-frequency 必须大于 0")
    return args


def escape_table_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


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

    annotations: list[str] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        result = model.pipeline([text for _, text in batch], tasks=["cws", "dep"])

        for words, dependency in zip(result.cws, result.dep, strict=True):
            pairs: list[str] = []
            for index, (head, label) in enumerate(
                zip(dependency["head"], dependency["label"], strict=True)
            ):
                if label != "ATT":
                    continue
                head_word = "ROOT" if head == 0 else words[head - 1]
                pairs.append(f"{words[index]} → {head_word}")
            annotations.append("；".join(pairs) if pairs else "无")

    if len(annotations) != len(records):
        raise RuntimeError(f"记录数不一致：输入 {len(records)}，输出 {len(annotations)}")

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
        "- 说明：结果为模型自动标注；领域短语和歧义句建议结合业务语义人工复核。",
        "",
        "| 原文件行号 | 原句 | 定语（修饰语 → 中心词） |",
        "|---:|---|---|",
    ]

    for (line_number, sentence), annotation in zip(records, annotations, strict=True):
        output_lines.append(
            f"| {line_number} | {escape_table_cell(sentence)} | "
            f"{escape_table_cell(annotation)} |"
        )

    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    if args.use_segmentation_words:
        print(
            f"已加载 SEGMENTATION_WORDS：{len(SEGMENTATION_WORDS)} 词，"
            f"freq={args.segmentation_word_frequency}"
        )
    print(f"已写入 {args.output}：{len(records)} 句")


if __name__ == "__main__":
    main()
