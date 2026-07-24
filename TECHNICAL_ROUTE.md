# 业务 RAG 定语提取技术路线

## 1. 项目目标

本项目从中文业务问题中提取对 RAG 检索有价值的定语（ATT），用于补充业务对象的限定信息。

最终结果重点保留：

- 等级：`A级 → 项目`
- 风险：`高风险 → 项目`
- 状态：`假设未关闭的 → 项目`
- 业务类型：`交付 → 项目`
- 完整原因从句：`产品质量导致的 → 网上事故`
- 完整关联从句：`设备增量相关的 → 收入`
- 业务专名：`NPX → 保障`、`EI → 项目`
- 指标构成：`管理 → 成熟度`

最终结果不保留：

- 时间内部关系：`2026年 → 5月`
- 数量和疑问成分：`多少 → 个`
- 指示和上下文数量：`这两个 → 事故`
- 已由维度抽取阶段删除的地域、组织、产品等维度
- 只有功能词、缺少业务内容的关系：`导致 → 项目`
- 语义改写或原句中不存在的标签

## 2. 核心原则

### 2.1 高召回与高精度分层

LTP 负责高召回地产生句法候选，Qwen 负责业务相关性判断，程序规则负责最终硬校验。

不要求单一模型同时完成分词、句法分析、业务判断和结果约束。

### 2.2 分析原始句，输出保留句

- 使用 `data/original_question.md` 的完整问题理解句法和业务语义。
- 使用 `data/dimension_extracted_question.md` 确定已经抽取的维度。
- 最终 ATT 的修饰语和中心词必须仍然存在于维度抽取后的句子中。

### 2.3 不做语义改写

`modifier_text`、`modifier_core` 和 `head_text` 必须逐字来自原句。

例如：

```text
原句：产品质量导致的网上事故
正确：产品质量导致的 → 网上事故
错误：事故原因=产品质量
```

### 2.4 完整从句与直接关系并存

- 普通复合名词保留逐层直接 ATT。
- “的”字原因、关联、状态从句保留完整连续文本。

例如：

```text
NPX保障项目：
NPX → 保障
保障 → 项目

由于物料供应原因导致的风险交付项目：
由于物料供应原因导致的 → 项目
交付 → 项目
```

## 3. 总体架构

```text
原始问题
   │
   ├── SEGMENTATION_WORDS 领域分词保护
   │
   ▼
LTP/base
   │
   ├── 中文分词
   ├── 依存句法
   └── 原始 ATT
   │
   ▼
完整定语候选重建
   │
   ├── token 字符区间对齐
   ├── “的”字定语子树合并
   ├── 原因/关联/状态从句重建
   └── “项目是由于……导致的”后置条件识别
   │
   ▼
LTP 完整候选 JSONL
   │
   ├── 对齐 dimension_extracted_question
   ├── 计算 excluded_dimensions
   └── 程序预删除时间、数量和已抽取维度
   │
   ▼
Qwen-plus v4
   │
   ├── keep：保留有效候选
   ├── drop：删除无效候选
   └── add：补充明确漏项
   │
   ▼
Schema 与硬规则校验
   │
   ├── 原句字符区间校验
   ├── 目标句字符区间校验
   ├── 业务专名保护
   ├── 指标构成保护
   ├── 补项形态校验
   └── 重复和内部冗余校验
   │
   ▼
最终业务 RAG ATT
```

## 4. 第一阶段：LTP 候选生成

核心代码：

- `extract_attributives_ltp.py`

### 4.1 领域分词保护

`SEGMENTATION_WORDS` 保存最小不可拆业务单元，例如：

```text
产品质量
物料供应
网上事故
高风险
NPX
EI
EHS
Facility
SmartCare
```

词典只用于保护分词，不直接决定 ATT 关系。

不应加入需要继续分析内部结构的完整长指标，例如：

```text
服务收入完成率
网络变更操作成功率
```

### 4.2 字符区间对齐

LTP 返回 token 和依存关系后，将每个 token 按顺序定位回原句，保存：

```json
{
  "text": "NPX",
  "start": 12,
  "end": 15,
  "head": 8,
  "label": "ATT"
}
```

后续所有结果都通过字符区间校验，防止模型改写或错位。

### 4.3 完整定语重建

对带“的”的 ATT 核心，基于依存子树向左收集主语、宾语、状语和补语：

```text
导致 → 项目
```

重建为：

```text
由于物料供应原因导致的 → 项目
```

重建过程中：

- 遇到 `由于`、`由`，从介词标记开始截取。
- 遇到标点停止跨句扩展。
- 在 `全球、运营商、业务、地区部` 等范围词处停止扩展。
- 删除 `这、两、多少、个` 等低价值成分。
- 所有完整修饰语都直接截取原句连续字符。

### 4.4 后置条件识别

LTP 不一定会把下面的原因条件标记成 ATT：

```text
项目是由于解决方案问题导致的
```

程序通过“实体 + 是 + 原因从句”的依存结构补充：

```text
由于解决方案问题导致的 → 项目
```

### 4.5 运行命令

```bash
python extract_attributives_ltp.py \
  data/original_question.md \
  output/original_question_attributives_ltp_reconstructed.md \
  --model LTP/base \
  --use-segmentation-words \
  --reconstruct-modifiers \
  --jsonl-output output/original_question_attributives_ltp_reconstructed.jsonl
```

### 4.6 第一阶段输出

- `output/original_question_attributives_ltp_reconstructed.md`
- `output/original_question_attributives_ltp_reconstructed.jsonl`

JSONL 保存：

- 原文件行号
- 原句
- token 和字符区间
- 依存关系
- LTP 原始 ATT
- 完整定语候选

## 5. 第二阶段：业务 RAG 筛选

核心代码：

- `filter_attributives_rag.py`

当前提示词版本：

```text
rag_candidate_filter_v4
```

当前模型：

```text
qwen-plus
```

### 5.1 数据对齐

按照原文件行号对齐：

- LTP JSONL 中的完整原句
- `dimension_extracted_question.md` 中的维度抽取后问题

通过字符序列差异计算 `excluded_dimensions`。

如果维度抽取后的行为空或文件末尾缺少对应行，按“该句已全部抽取”处理，最终 ATT 为空，不与下一行错位。

### 5.2 程序预过滤

调用 Qwen 前，程序先删除：

- 不在目标句中的修饰语或中心词
- 时间表达
- 数量和疑问成分
- 只有功能词的候选
- 已抽取维度覆盖的候选

Qwen 只处理字符区间合法的剩余候选。

### 5.3 Qwen 的职责

Qwen 对每个候选执行：

```text
keep：保留为业务 RAG ATT
drop：删除低价值、错误或冗余候选
add：补充 LTP 明确漏掉的直接 ATT
```

模型不能使用 `excluded_dimension` 删除已经通过预过滤的候选。

`keep` 必须使用：

```text
business_constraint
```

### 5.4 补项约束

LLM 补项只允许两类：

1. 单个 LTP token 修饰最小名词中心词。
2. 完整的原因、关联或状态定语从句。

允许：

```text
EI → 项目
EHS → 管理
产品质量导致的 → 网上事故
```

禁止：

```text
销毛率 → 销毛率
解决方案问题 → 导致的
高危网络变更操作 → 操作
业务的 → 成本率
```

补项还必须满足：

- 修饰语与中心词不能相同。
- 中心词不能包含在修饰语中。
- 修饰语和中心词之间不能跨越谓语。
- 不能添加完整从句内部的冗余关系。
- 不能添加已有候选的重复关系。
- 不能使用泛化范围词作为补项。

### 5.5 业务专名保护

以下词只要仍存在于目标句中，就不能被模型错误删除：

```text
NPX、EHS、EI、ITS、NIS、SEC、AMS、MBB、FBB、
IT、DC、5G、P3、H1、TOP3、Facility、SmartCare
```

例如：

```text
NPX保障项目
→ NPX → 保障
→ 保障 → 项目
```

### 5.6 指标构成保护

中心词以“率”或“度”结尾时，业务构成关系受到程序保护，例如：

```text
收入 → 完成率
管理 → 成熟度
风险 → 超期未关闭率
```

### 5.7 运行命令

运行前需要激活包含依赖的 Python 环境，并在 `.env` 中配置：

```text
DASHSCOPE_API_KEY=...
```

执行：

```bash
python filter_attributives_rag.py \
  output/original_question_attributives_ltp_reconstructed.jsonl \
  data/dimension_extracted_question.md \
  output/original_question_attributives_rag_filtered.md \
  --jsonl-output output/original_question_attributives_rag_filtered.jsonl \
  --batch-size 8
```

### 5.8 缓存与断点续跑

每个成功批次都会写入：

```text
output/.cache/
```

缓存签名包含：

- 提示词版本
- 模型
- 原句
- 目标句
- 已排除维度
- 候选内容

输入或提示词版本变化后，旧缓存不会被错误复用。

## 6. 最终输出结构

Markdown：

```text
原文件行号
原始问题
维度抽取后问题
已排除维度
最终 ATT
```

JSONL 额外保存：

- 候选数量
- keep/drop 决策
- 接受的 LLM 补项
- 被硬规则拒绝的补项及原因
- 最终 ATT
- 原始句和目标句字符区间
- 候选来源

候选来源包括：

```text
ltp_token
ltp_subtree
ltp_copular_clause
llm_addition
```

## 7. 当前全量结果

数据规模：

```text
389 句
```

v4 全量结果：

```text
最终 ATT：452 条
存在有效 ATT 的句子：284 条
接受 LLM 补项：27 条
硬校验拒绝补项：58 条
维度抽取后为空：29 句
字符区间错误：0
```

API 用量：

```text
输入：124,200 tokens
输出：30,970 tokens
```

正式结果：

- `output/original_question_attributives_rag_filtered.md`
- `output/original_question_attributives_rag_filtered.jsonl`

## 8. 典型结果

```text
由于物料供应原因导致的 → 项目
产品质量导致的 → 网上事故
未及时恢复的 → 网上事故
设备增量相关的 → 收入
二级事故的 → 根因
假设未关闭的 → 项目
A级 → 项目
EI → 项目
NPX → 保障
保障 → 项目
EHS → 管理
管理 → 成熟度
SmartCare → 场景
```

## 9. 已知边界

### 9.1 依存句法误差

LTP 仍可能把业务复合词分析为主谓、动宾或兼语关系，导致漏掉直接 ATT。

当前通过：

- 领域分词
- 完整从句重建
- Qwen 补项
- 业务专名保护

降低影响，但不能完全消除。

### 9.2 维度抽取质量会影响最终 ATT

如果 `dimension_extracted_question.md` 删除过多内容，程序会按要求禁止这些字符出现在最终 ATT 中。

因此维度抽取阶段与 ATT 阶段需要共同评估。

### 9.3 LLM 业务判断仍可能波动

提示词、Schema 和硬规则已经限制输出，但 `keep/drop` 仍包含模型判断。

缓存可保证同一批结果稳定复用；提示词升级时应重新运行代表性测试集。

### 9.4 尚未建立人工金标准

当前质量主要通过关键样例和程序一致性校验评估。

程序校验能够保证：

- 不改写
- 不越过排除维度
- 字符区间正确
- 补项形态合法

但不能完全替代业务专家对“是否值得用于 RAG”的判断。

## 10. 建议的下一阶段

建立 50 至 100 句人工标注评测集，覆盖：

- 原因从句
- 关联从句
- 状态定语
- 等级和风险
- 业务缩写
- 复合指标
- 已抽取维度
- 省略和歧义句

建议计算：

```text
候选召回率
最终 ATT 精确率
最终 ATT 召回率
完全匹配率
错误维度泄漏率
LLM 补项接受率
```

在人工评测稳定后，再将最终 ATT 作为 RAG 元数据字段写入索引。
