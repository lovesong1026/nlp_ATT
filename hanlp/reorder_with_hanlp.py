from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

from reordering_core import (
    Analysis,
    candidate_sentences,
    make_plan,
    read_records,
    write_markdown,
)


ROOT = Path(__file__).resolve().parent.parent
HANLP_DIR = ROOT / "hanlp"
DEFAULT_INPUT = ROOT / "data" / "original_question.md"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "original_question_reordered_hanlp.md"
DEFAULT_ACCEPTED_OUTPUT = (
    Path(__file__).resolve().parent / "original_question_reordered_hanlp_accepted.md"
)
DEFAULT_MODEL_HOME = HANLP_DIR / "models"
SEGMENTATION_WORDS = {
    "运营商BG",
    "中东中亚",
    "地区部",
    "小国小网",
    "变更倒回",
    "高风险",
    "中风险",
    "低风险",
    "S级",
    "A级",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用HanLP检测并校验后置因果定语语序重排。")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--accepted-output",
        type=Path,
        default=DEFAULT_ACCEPTED_OUTPUT,
        help="仅保存自动接受重排结果的Markdown文件",
    )
    parser.add_argument("--model-home", type=Path, default=DEFAULT_MODEL_HOME)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size必须大于0")
    return args


def batches(model: Any, values: list[Any], batch_size: int) -> list[Any]:
    output: list[Any] = []
    for start in range(0, len(values), batch_size):
        output.extend(model(values[start : start + batch_size]))
    return output


def release(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def parse_srl(frames: list[Any]) -> tuple[list[int], dict[int, list[tuple[int, int, str]]]]:
    predicates: list[int] = []
    arguments: dict[int, list[tuple[int, int, str]]] = {}
    for frame in frames:
        predicate: int | None = None
        roles: list[tuple[int, int, str]] = []
        for text, label, start, end in frame:
            if label == "PRED":
                predicate = int(start)
            else:
                roles.append((int(start), int(end) - 1, str(label)))
        if predicate is not None:
            predicates.append(predicate)
            arguments[predicate] = roles
    return predicates, arguments


def load_analyses(sentences: list[str], model_home: Path, batch_size: int) -> list[Analysis]:
    if not sentences:
        return []
    os.environ["HANLP_HOME"] = str(model_home.resolve())
    import hanlp

    # 任务权重统一平铺在 ``hanlp/models/<模型目录名>``，不再依赖
    # HanLP按下载URL生成的 thirdparty 缓存层级。
    specs = {
        "tok": model_home / "ctb9_tok_electra_base_crf_20220426_161255",
        "pos": model_home / "pos_ctb_electra_small_20220215_111944",
        "dep": model_home / "ctb9_dep_electra_small_20220216_100306",
        "srl": model_home / "cpb3_electra_small_crf_has_transform_20220218_135910",
        "con": model_home / "ctb9_con_electra_small_20220215_230116",
    }
    missing = [path.as_posix() for path in specs.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("缺少HanLP任务模型：" + "，".join(missing))

    tokenizer = hanlp.load(str(specs["tok"]))
    tokenizer.dict_force = SEGMENTATION_WORDS
    tokens = batches(tokenizer, sentences, batch_size)
    release(tokenizer)

    results: dict[str, list[Any]] = {}
    for task in ("pos", "dep", "srl", "con"):
        model = hanlp.load(str(specs[task]))
        results[task] = batches(model, tokens, batch_size)
        release(model)
        print(f"HanLP {task}：{len(results[task])}/{len(sentences)}")

    analyses: list[Analysis] = []
    for words, pos, dep, srl, con in zip(
        tokens,
        results["pos"],
        results["dep"],
        results["srl"],
        results["con"],
        strict=True,
    ):
        predicates, arguments = parse_srl(srl)
        analyses.append(
            Analysis(
                words=list(words),
                pos=list(pos),
                heads=[int(token["head"]) for token in dep],
                labels=[str(token["deprel"]) for token in dep],
                srl=srl,
                predicates=predicates,
                argument_spans=arguments,
                constituency=str(con),
            )
        )
    return analyses


def write_accepted_results(
    path: Path,
    input_path: Path,
    records: list[tuple[int, str]],
    plans: list[Any],
    decisions: list[Any],
) -> int:
    rows = [
        (line_number, sentence, plan, decision)
        for (line_number, sentence), plan, decision in zip(records, plans, decisions, strict=True)
        if decision.status == "accepted"
    ]
    lines = [
        f"# {input_path.stem} HanLP 自动接受的重排数据",
        "",
        f"- 来源：`{input_path.as_posix()}`。",
        "- 仅包含通过二次 DEP + SRL 校验的重排结果。",
        f"- 共 {len(rows)} 条。",
        "",
        "| 原文件行号 | 原句 | 命中规则 | 重排结果 | 验证依据 |",
        "|---:|---|---|---|---|",
    ]
    for line_number, sentence, plan, decision in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(line_number),
                    sentence.replace("|", "\\|"),
                    plan.rule_type.replace("|", "\\|"),
                    decision.output.replace("|", "\\|"),
                    decision.evidence.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    originals = load_analyses(
        [sentence for _line, sentence in records], args.model_home, args.batch_size
    )
    plans = [make_plan(analysis, "hanlp") for analysis in originals]
    candidate_indices, candidates = candidate_sentences(plans)
    candidate_values = load_analyses(candidates, args.model_home, args.batch_size)
    candidate_analyses = dict(zip(candidate_indices, candidate_values, strict=True))
    generated, accepted, decisions = write_markdown(
        args.output,
        args.input,
        "hanlp",
        "`CTB9_TOK_ELECTRA_BASE_CRF + CTB9_POS_ELECTRA_SMALL + "
        "CTB9_DEP_ELECTRA_SMALL + CPB3_SRL_ELECTRA_SMALL + CTB9_CON_ELECTRA_SMALL`",
        records,
        originals,
        plans,
        candidate_analyses,
    )
    accepted_rows = write_accepted_results(
        args.accepted_output, args.input, records, plans, decisions
    )
    print(
        f"HanLP完成：{len(records)}句，候选{generated}句，接受{accepted}句；"
        f"完整结果：{args.output}；精简结果：{args.accepted_output}（{accepted_rows}条）"
    )


if __name__ == "__main__":
    main()
