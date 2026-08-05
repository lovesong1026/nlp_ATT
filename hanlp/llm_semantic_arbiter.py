"""以 Qwen 复核 HanLP 低置信语义候选，不覆盖规则结果。

职责边界：只处理未知指标、实体碎片、维度歧义等低置信记录，产出可审计
的修订建议。所有建议必须逐字出现在原问题中，且不能与维度删除区间重叠。
最终是否采用仍由下游 Schema 决定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_INPUT = ROOT / "original_question_reordered_dep_att.md"
DEFAULT_OUTPUT = ROOT / "llm_semantic_arbiter.md"
DEFAULT_DIMENSIONS = PROJECT_ROOT / "data" / "dimension_extracted_question.md"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROMPT_VERSION = "semantic-arbiter-v1"
QUESTION_DIMENSION_RE = re.compile(r"(?:哪个|哪些|哪几个|哪)\S{0,12}?的")
COUNT_OR_VALUE_RE = re.compile(r"(?:多少|几|数量|次数|总数|最多|最少|最高|最低|率|成熟度)")


@dataclass(frozen=True)
class Row:
    line_id: int
    original: str
    dimensionless: str
    intent: str
    entities: str
    indicators: str
    dep_att: str
    conditions: str
    object_metric: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 Qwen 仅复核 HanLP 的低置信实体、指标和条件候选。"
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimensions", type=Path, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--schema", type=Path, default=None,
                        help="可选JSON：{\"entities\": [...], \"indicators\": [...]}。")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--line-ids", default=None,
                        help="逗号分隔的原文件行号，例如 172,226,345。")
    parser.add_argument("--all", action="store_true",
                        help="审查全部非空维度结果；默认只审查低置信记录。")
    parser.add_argument("--dry-run", action="store_true", help="只生成待审查清单，不调用API。")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument(
        "--merge-results", nargs="+", type=Path, default=None,
        help="合并已有裁决 Markdown（断点批处理后使用）；不调用 API。",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于0")
    return args


def split_markdown_row(line: str) -> list[str]:
    """解析含转义竖线的Markdown表格行。"""
    cells: list[str] = []
    value: list[str] = []
    escaped = False
    for char in line.strip()[1:-1]:
        if escaped:
            value.append(char)
            escaped = False
        elif char == "\\":
            value.append(char)
            escaped = True
        elif char == "|":
            cells.append("".join(value).strip())
            value = []
        else:
            value.append(char)
    cells.append("".join(value).strip())
    return cells


def read_rows(path: Path) -> list[Row]:
    headers: list[str] | None = None
    rows: list[Row] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith("|"):
            continue
        cells = split_markdown_row(raw_line)
        if headers is None:
            if cells and cells[0] == "原文件行号":
                headers = cells
            continue
        if cells and re.fullmatch(r":?-+:?", cells[0]):
            continue
        if len(cells) != len(headers):
            raise ValueError(f"Markdown列数不一致：{raw_line}")
        values = dict(zip(headers, cells, strict=True))
        rows.append(
            Row(
                line_id=int(values["原文件行号"]),
                original=values["原始问题"],
                dimensionless=values["去时间、维度的问题"],
                intent=values["查询意图"],
                entities=values["业务实体候选"],
                indicators=values["指标候选"],
                dep_att=values["去时间、维度后的 DEP 原子ATT关系"],
                conditions=values["修饰 → 实体"],
                object_metric=values["对象 → 指标"],
            )
        )
    if headers is None:
        raise ValueError(f"未找到Markdown表头：{path}")
    return rows


def read_dimension_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()[1:]]


def read_schema(path: Path | None) -> dict[str, set[str]]:
    """读取可选的业务 Schema；缺省时只做原文与维度硬校验。"""
    if path is None:
        return {"entities": set(), "indicators": set()}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Schema 读取失败：{path}（{exc}）") from exc
    if not isinstance(raw, dict):
        raise ValueError("Schema 顶层必须是对象")

    schema: dict[str, set[str]] = {}
    for field in ("entities", "indicators"):
        values = raw.get(field, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"Schema.{field} 必须是字符串数组")
        schema[field] = set(values)
    return schema


def excluded_spans(original: str, dimension_sentence: str) -> list[tuple[int, int]]:
    """得到原句相对维度提取句被删除的字符区间。"""
    if not dimension_sentence:
        return []
    return [
        (start, end)
        for tag, start, end, _other_start, _other_end in SequenceMatcher(
            None, original, dimension_sentence
        ).get_opcodes()
        if tag in {"delete", "replace"} and start < end
    ]


def overlaps_excluded(text: str, original: str, spans: list[tuple[int, int]]) -> bool:
    """文本所有出现位置均与维度区间重叠，才视为不可用。"""
    if not text or text == "—":
        return False
    starts = [match.start() for match in re.finditer(re.escape(text), original)]
    if not starts:
        return True
    return all(
        any(start < right and start + len(text) > left for left, right in spans)
        for start in starts
    )


def low_confidence_reasons(row: Row) -> list[str]:
    if row.dimensionless == "—":
        return []
    reasons: list[str] = []
    if row.entities == "—":
        reasons.append("实体缺失")
    if row.indicators == "—" and (
        row.intent in {"数量查询", "指标值查询"}
        or bool(COUNT_OR_VALUE_RE.search(row.dimensionless))
    ):
        reasons.append("指标缺失")
    if "；" in row.entities:
        reasons.append("实体边界冲突")
    if QUESTION_DIMENSION_RE.search(row.dimensionless):
        reasons.append("查询维度可能未过滤")
    if any(term in row.original for term in ("销毛率", "贡毛率", "利润率")):
        reasons.append("未知指标别名")
    return list(dict.fromkeys(reasons))


def parse_line_ids(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(part) for part in value.split(",") if part.strip()}


def prompt_payload(
    row: Row, reasons: list[str], spans: list[tuple[int, int]], schema: dict[str, set[str]]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "line_id": row.line_id,
        "original_question": row.original,
        "dimensionless_question": row.dimensionless,
        "excluded_dimension_spans": [{"start": start, "end": end} for start, end in spans],
        "rule_result": {
            "intent": row.intent,
            "entities": row.entities,
            "indicators": row.indicators,
            "conditions": row.conditions,
            "object_metric": row.object_metric,
            "dep_att": row.dep_att,
        },
        "review_reasons": reasons,
    }
    # Schema 可能很大，只在显式传入时提供候选集合，最终仍由本地硬校验裁决。
    if schema["entities"] or schema["indicators"]:
        payload["schema"] = {
            "entities": sorted(schema["entities"]),
            "indicators": sorted(schema["indicators"]),
        }
    return payload


SYSTEM_PROMPT = """你是问数RAG语义裁决器，不是自由抽取器。
只复核输入中列出的低置信问题。原始问题是唯一事实来源；不得改写、补造或使用
原句中不存在的文本。时间、地域、组织等已排除维度不得进入实体、指标或条件。
保留规则结果，除非有明确理由替换。若缺少业务Schema而无法确认指标别名，必须
abstain。返回严格JSON：
{
  "decision": "keep|replace|abstain",
  "entities": ["原句连续片段"],
  "indicators": ["原句连续片段"],
  "conditions": [{"modifier":"原句连续片段", "entity":"原句连续片段"}],
  "reason": "不超过60字"
}
"""


def validate_response(
    value: dict[str, Any], row: Row, spans: list[tuple[int, int]], schema: dict[str, set[str]]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    decision = value.get("decision")
    if decision not in {"keep", "replace", "abstain"}:
        errors.append("非法decision，已改为abstain")
        decision = "abstain"

    def validate_texts(items: Any, field: str) -> list[str]:
        if not isinstance(items, list):
            errors.append(f"{field}不是列表")
            return []
        accepted: list[str] = []
        for item in items:
            if not isinstance(item, str) or not item or item not in row.original:
                errors.append(f"{field}含非原文片段：{item!r}")
                continue
            if overlaps_excluded(item, row.original, spans):
                errors.append(f"{field}与维度区间重叠：{item}")
                continue
            accepted.append(item)
        return list(dict.fromkeys(accepted))

    entities = validate_texts(value.get("entities", []), "entities")
    indicators = validate_texts(value.get("indicators", []), "indicators")
    for field, items in (("entities", entities), ("indicators", indicators)):
        allowed = schema[field]
        if allowed:
            for item in items:
                if item not in allowed:
                    errors.append(f"{field}未命中Schema：{item}")
    conditions: list[dict[str, str]] = []
    raw_conditions = value.get("conditions", [])
    if not isinstance(raw_conditions, list):
        errors.append("conditions不是列表")
    else:
        for item in raw_conditions:
            if not isinstance(item, dict):
                errors.append("conditions含非对象")
                continue
            modifier, entity = item.get("modifier"), item.get("entity")
            if not isinstance(modifier, str) or not isinstance(entity, str):
                errors.append("conditions字段类型错误")
                continue
            if modifier not in row.original or entity not in row.original:
                errors.append(f"conditions含非原文片段：{item}")
                continue
            if overlaps_excluded(modifier, row.original, spans) or overlaps_excluded(entity, row.original, spans):
                errors.append(f"conditions与维度区间重叠：{item}")
                continue
            conditions.append({"modifier": modifier, "entity": entity})
    return {
        "decision": decision,
        "entities": entities,
        "indicators": indicators,
        "conditions": conditions,
        "reason": str(value.get("reason", ""))[:120],
    }, errors


def cache_key(payload: dict[str, Any], model: str) -> str:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{PROMPT_VERSION}\n{model}\n{content}".encode()).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    values: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            result = item["result"]
            # 网络错误不是语义裁决结果；下次运行应重试而不是永久命中缓存。
            if str(result.get("reason", "")).startswith("调用失败："):
                continue
            values[item["key"]] = result
        except (json.JSONDecodeError, KeyError):
            continue
    return values


def call_llm(client: OpenAI, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except (OpenAIError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    assert last_error is not None
    raise last_error


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def merge_result_files(paths: list[Path], output: Path) -> None:
    """按原文件行号合并断点批处理结果，后出现的相同行号覆盖前者。"""
    rows: dict[int, str] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| "):
                continue
            cells = split_markdown_row(line)
            if not cells or not cells[0].isdigit():
                continue
            rows[int(cells[0])] = line
    lines = [
        "# LLM 低置信语义裁决建议",
        "",
        "- 由断点批处理结果合并；提示词版本：`semantic-arbiter-v1`。",
        "- 本文件是建议，不覆盖 HanLP 正式结果；需经业务 Schema 校验后才可采用。",
        "- 硬校验：建议中的实体、指标、条件必须逐字出现于原问题，且不能与维度删除区间重叠。",
        f"- 待审查：{len(rows)}条；模式：qwen-plus。",
        "",
        "| 原文件行号 | 原始问题 | 触发原因 | 规则实体 | 规则指标 | 规则条件 | LLM建议 | 校验拒绝原因 |",
        "|---:|---|---|---|---|---|---|---|",
        *(rows[line_id] for line_id in sorted(rows)),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"已合并 {len(rows)} 条裁决：{output}")


def main() -> None:
    args = parse_args()
    if args.merge_results:
        merge_result_files(args.merge_results, args.output)
        return
    rows = read_rows(args.input)
    dimension_lines = read_dimension_lines(args.dimensions)
    schema = read_schema(args.schema)
    selected_ids = parse_line_ids(args.line_ids)
    selected: list[tuple[Row, list[str], list[tuple[int, int]]]] = []
    for row in rows:
        if selected_ids is not None and row.line_id not in selected_ids:
            continue
        reasons = low_confidence_reasons(row)
        if not args.all and not reasons:
            continue
        dimension = dimension_lines[row.line_id - 2] if row.line_id - 2 < len(dimension_lines) else ""
        selected.append((row, reasons, excluded_spans(row.original, dimension)))
    if args.limit is not None:
        selected = selected[: args.limit]

    cache_path = args.cache or args.output.parent / ".cache" / f"{args.output.stem}.{args.model}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    client: OpenAI | None = None
    if not args.dry_run:
        load_dotenv(args.env_file)
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(f"未在 {args.env_file} 中找到 DASHSCOPE_API_KEY")
        client = OpenAI(api_key=api_key, base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))

    lines = [
        "# LLM 低置信语义裁决建议",
        "",
        f"- 输入：`{args.input.as_posix()}`；提示词版本：`{PROMPT_VERSION}`。",
        "- 本文件是建议，不覆盖 HanLP 正式结果；需经业务 Schema 校验后才可采用。",
        "- 硬校验：建议中的实体、指标、条件必须逐字出现于原问题，且不能与维度删除区间重叠。",
        f"- 待审查：{len(selected)}条；模式：{'dry-run' if args.dry_run else args.model}。",
        "",
        "| 原文件行号 | 原始问题 | 触发原因 | 规则实体 | 规则指标 | 规则条件 | LLM建议 | 校验拒绝原因 |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    # 每条结果与缓存都立即写盘：长批任务可安全中断并从缓存续跑。
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for index, (row, reasons, spans) in enumerate(selected, start=1):
        payload = prompt_payload(row, reasons, spans, schema)
        key = cache_key(payload, args.model)
        errors: list[str] = []
        if args.dry_run:
            result = {"decision": "pending", "entities": [], "indicators": [], "conditions": [], "reason": "dry-run"}
        elif key in cache:
            result = cache[key]
        else:
            try:
                assert client is not None
                raw = call_llm(client, args.model, payload)
                result, errors = validate_response(raw, row, spans, schema)
            except (OpenAIError, json.JSONDecodeError, ValueError) as exc:
                result = {"decision": "abstain", "entities": [], "indicators": [], "conditions": [], "reason": f"调用失败：{type(exc).__name__}"}
                errors.append(str(exc)[:160])
            with cache_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
        suggestion = json.dumps(result, ensure_ascii=False)
        output_row = (
            f"| {row.line_id} | {escape(row.original)} | {escape('；'.join(reasons) or '全量审查')} | "
            f"{escape(row.entities)} | {escape(row.indicators)} | {escape(row.conditions)} | "
            f"{escape(suggestion)} | {escape('；'.join(errors) or '—')} |"
        )
        with args.output.open("a", encoding="utf-8") as file:
            file.write(output_row + "\n")
        print(f"已处理 {index}/{len(selected)}：原文件行号 {row.line_id}", flush=True)
    print(f"LLM语义裁决建议已写入：{args.output}（{len(selected)}条）")


if __name__ == "__main__":
    main()
