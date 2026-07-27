# 3. POS 辅助 DEP 定语提取

本阶段在第2阶段自定义分词词典的基础上，只运行：

```text
cws + pos + dep
```

目标是验证 POS 如何辅助 DEP/ATT 提取业务定语，并可选地结合
`dimension_extracted_question.md` 过滤已经抽取的维度。当前不加入
LLM、SRL、Schema 映射或问句改写。

## 当前配置

- 模型：`LTP/base`
- 自定义词典：66词
- 自定义词频：`freq=2`
- 输入：`data/original_question.md`
- 输出：`original_question_attributives_pos_dep.md`

## POS 辅助规则

1. DEP 的 `ATT` 是定语关系锚点。
2. POS 为 `r/m/q/nt` 的疑问、数量和时间修饰默认过滤。
3. `2026年 → 5月` 一类日历内部修饰过滤。
4. 自定义词典中的业务词优先保留，即使 LTP 将其标成 `v/m`。
5. 动词性 ATT 如果带结构助词“的”，从原句重建完整定语从句。
6. 输出只截取原句，不进行语义改写。
7. ATT 错挂到“个、次、张”等量词时，沿依存中心提升到内容词。
8. 对漏标的等级、风险、“交付”、网络质量及英文数字专名，只在右侧局部窗口中恢复到明确业务实体。

## 维度过滤

通过 `--dimension-input` 传入维度提取后的问题时：

1. 两个文件按物理行号对齐，空行也占据对应位置。
2. 维度后问题必须是原句忽略空白后的纯删除结果，出现替换时立即报错。
3. 代码记录原句中被删除维度的字符区间。
4. 定语或中心词的字符区间只要与已删除维度重叠，就过滤整条候选。
5. 维度文件末尾缺失的行按空结果处理，即整句候选全部过滤。

模型仍分析完整原句，维度后问题只参与最终候选过滤，不参与 CWS、POS
或 DEP 推理。

例如：

```text
原始 ATT：导致/v → 网上事故/nl
最终结果：产品质量导致的 → 网上事故
```

## 运行命令

```bash
conda run -n minimind python \
  "3.extract_attributives_pos_dep/extract_attributives_pos_dep.py" \
  "data/original_question.md" \
  "3.extract_attributives_pos_dep/original_question_attributives_pos_dep.md"
```

启用维度过滤：

```bash
conda run -n minimind python \
  "3.extract_attributives_pos_dep/extract_attributives_pos_dep.py" \
  "data/original_question.md" \
  "3.extract_attributives_pos_dep/original_question_attributives_pos_dep_dimension_filtered.md" \
  --dimension-input "data/dimension_extracted_question.md"
```

## 输出列

- 原始问题
- CWS 分词
- POS 词性
- DEP 原始 ATT
- POS 辅助判定及原因
- 最终完整定语

启用维度过滤后还会输出：

- 维度后问题
- 已删除维度片段
- 维度过滤前定语
- 维度过滤判定
- 最终定语（去维度）
