# 4. SRL辅助动词定语恢复

第四阶段保持第三阶段的原子ATT基线不变，只对动词性ATT增加SRL解释和
完整片段恢复。

```text
cws + pos + dep + srl
```

## 职责划分

- DEP：确定“哪个动词修饰哪个中心词”。
- POS：限制候选为动词性ATT。
- 自定义词典中的中心词不因 POS 被误标为 `m/q` 而过滤。
- SRL：匹配同一token谓词，提供A0/A1/A2及原因、否定、方式等论元。
- “的”：只作为显式定语边界和置信度信号，不是恢复的必要条件。
- 原句字符区间：决定最终文本，禁止补词、调序和语义改写。

第四阶段最后一列是可直接读取的完整结果：

1. 先继承第三阶段经过POS过滤的全部原子ATT。
2. 非动词定语（如`S级 → 项目`）原样保留。
3. 动词定语SRL恢复成功时，用完整定语替换对应的短原子关系。
4. SRL恢复失败时，保留原始动词ATT，不丢失关系。

## 恢复条件

1. DEP标签必须为`ATT`。
2. 修饰词POS必须以`v`开头。
3. DEP中心词不能是疑问词、数量词、量词或标点。
4. SRL必须识别同一个token位置的谓词。
5. 至少存在一个位于中心词之前的有效SRL论元。
6. 恢复片段不能跨越标点。
7. 最终定语必须是原句中的连续字符片段。

第一版纳入的SRL论元：

```text
A0 A1 A2 A3 A4 A5
ARGM-PRP ARGM-CAU ARGM-ADV ARGM-NEG
ARGM-MNR ARGM-DIR ARGM-EXT
```

`ARGM-TMP`和`ARGM-LOC`暂不并入定语，避免把时间和范围维度重新带入。

## 置信度

- `high`：存在显式“的”，并且SRL论元覆盖DEP中心词。
- `medium`：存在中心词证据但没有“的”，或存在“的”但SRL未覆盖中心词。
- `low`：SRL提供了修饰论元，但既没有“的”也没有中心词证据。
- `unresolved`：SRL谓词缺失、没有有效修饰论元或恢复区间不合法。

只有`high`和`medium`进入“接受的动词定语”。`low`只显示在
“SRL连续候选”中，便于人工审计，不计为成功恢复。

SRL缺失时只保留DEP原子ATT和失败状态，不启用DEP子树fallback。这样可以
单独测量SRL带来的真实覆盖率和准确率。

## 示例

```text
DEP：导致 → 项目
SRL：ARGM-PRP=由于物料供应原因；A1=项目
恢复：由于物料供应原因导致的 → 项目
置信度：high
```

```text
DEP：占 → 占比
SRL：A0=服务订货；A2=运营商BG总订货；A1=占比
恢复：服务订货占运营商BG总订货的 → 占比
置信度：high
```

如果SRL没有识别“导致”：

```text
DEP：导致 → 网上事故
恢复：无
状态：srl_predicate_missing
```

## 运行

```bash
conda run -n minimind python \
  "4.recover_verbal_attributives_srl/extract_verbal_attributives_srl.py" \
  "data/original_question.md" \
  "4.recover_verbal_attributives_srl/original_question_verbal_attributives_srl.md"
```

共享词典默认为项目根目录的：

```text
segmentation_words.txt
```

也可以通过以下参数切换：

```text
--segmentation-dictionary path/to/dictionary.txt
```

## 输出

- 原文件行号
- 原句
- CWS分词
- POS词性
- 第三阶段原子ATT
- DEP动词ATT
- SRL证据
- 恢复判定
- SRL连续候选
- 接受的动词定语
- 第四阶段完整结果
