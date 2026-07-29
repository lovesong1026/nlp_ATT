# 3. POS + DEP 原子定中关系提取

第三阶段只建立一个可解释的句法基线：

```text
cws（分词前置）+ pos + dep
```

## 本阶段做什么

1. 使用项目根目录的 `segmentation_words.txt` 干预 LTP 分词。
2. 运行 `LTP/base` 的 `cws + pos + dep`。
3. 只读取 DEP 标签为 `ATT` 的原子定中关系。
4. 使用 POS 和词面规则过滤时间、数量、疑问、指示和结构词等明显噪声。
   自定义词典中的中心词不因 POS 被误标为 `m/q` 而过滤。
5. 同时输出原始 ATT、POS 判定和过滤后的原子 ATT。

## 本阶段不做什么

- 不从原句重建完整定语。
- 不修正或提升 DEP 中心词。
- 不在局部窗口补充 LTP 漏掉的关系。
- 不补充系词后置定语从句。
- 不过滤 `dimension_extracted_question.md` 中的维度。
- 不合并 `A → B，B → C` 或 `A → C，B → C`。
- 不使用 LLM、SRL、NER、SDP、Schema 或语义改写。

因此，如果 DEP 输出：

```text
导致 → 网上事故
```

第三阶段会保留该原子关系，不会改成：

```text
产品质量导致的 → 网上事故
```

完整定语重建、维度过滤和关系合并应放在后续独立阶段，便于分别评测。

## 当前配置

- 模型：`LTP/base`
- 共享词典：项目根目录 `segmentation_words.txt`，当前15词
- 自定义词频：`freq=2`
- 输入：`data/original_question.md`
- 输出：`original_question_attributives_pos_dep.md`

## 运行命令

```bash
conda run -n minimind python \
  "3.extract_attributives_pos_dep/extract_attributives_pos_dep.py" \
  "data/original_question.md" \
  "3.extract_attributives_pos_dep/original_question_attributives_pos_dep.md"
```

可通过以下参数切换词典：

```text
--segmentation-dictionary path/to/dictionary.txt
```

词典使用 UTF-8 编码，每行一个词；空行和以 `#` 开头的注释行会被忽略。

## 输出列

- 原文件行号
- 原句
- CWS分词
- POS词性
- DEP原始ATT
- POS判定
- 最终原子ATT
