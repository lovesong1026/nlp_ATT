# 业务 RAG 定语抽取技术路线

## 1. 项目目标

本项目从中文业务问句中提取对 RAG 检索有价值的完整定语关系：

```text
modifier_text → head_text
```

其中：

- `modifier_text`：能够缩小检索范围的状态、原因、关联、动作、属性、
  领属或分类条件；
- `head_text`：承接定语、表示查询对象的业务实体短语；
- 两端都必须是原句中的连续片段，不补词、不调序、不做同义改写；
- 已由上游维度抽取删除的地域、组织、产品等内容，不得重新进入结果。

示例：

```text
产品质量导致的 → 网上事故
假设未关闭的 → A级交付EI项目
由于物料供应问题导致的 → 风险交付项目
```

以下内容通常不作为 RAG 定语：

```text
2026年 → 5月
多少 → 个
这两个 → 二级事故
导致 → 网上事故
```

## 2. 核心定义

### 2.1 原子 ATT

LTP 依存句法中标签为 `ATT` 的直接定中边：

```text
导致 → 事故
交付 → 项目
```

原子 ATT 可解释、召回高，但不一定具有完整业务语义。

### 2.2 完整业务定语

能够独立表达业务限制条件的原句连续片段：

```text
产品质量导致的 → 网上事故
```

它可能由多个 DEP/SRL 片段组合而来，也可能是 LLM 从原句补充的遗漏。

### 2.3 业务实体

用户实际查询、计数、列举或判断的业务对象，也是最终关系的中心词：

```text
中风险 → 交付项目
```

其中 `交付项目` 是业务实体，`中风险` 是业务定语。

业务实体不等同于传统 NER 中的人名、地名或机构名。当前实现要求
`head_text` 必须逐字等于模型识别的某个业务实体。

### 2.4 排除维度

`data/dimension_extracted_question.md` 是
`data/original_question.md` 经维度抽取后的对应结果。程序通过二者的字符
差异，将已删除内容映射成原句中的禁止区间。

第五阶段仍分析完整原句，但最终定语和中心词不能与禁止区间重叠。

## 3. 当前总体架构

```text
data/original_question.md
        │
        ├────────────────────────────────────────────┐
        │                                            │
        ▼                                            ▼
第1阶段：LTP七任务基线                    dimension_extracted_question.md
cws/pos/ner/srl/dep/sdp/sdpg                       │
        │                                            │
        ▼                                            │
第2阶段：加入最小分词词典                           │
验证词典对七任务的连锁影响                           │
        │                                            │
        ▼                                            │
第3阶段：DEP 原始事实                               │
保存完整依存图和全部原始 ATT                         │
        │                                            │
        ▼                                            │
第4阶段：POS + DEP + SDP + SRL                      │
筛选、纠错并补召回原子 ATT                           │
        │                                            │
        └──────────────────┬─────────────────────────┘
                           ▼
第5阶段：qwen-plus
识别业务实体，筛选、修剪、合并、补漏
                           │
                           ▼
Schema + 原文跨度 + 维度区间硬校验
                           │
                           ▼
可用于业务 RAG 的完整定语关系
```

总体原则：

1. 原句是唯一事实来源。
2. LTP、POS、DEP、SRL 只提供可错、可缺失的候选证据。
3. LLM 负责业务语义裁决，不负责创造原句中不存在的文本。
4. 程序负责确定性约束，不能把所有质量判断交给提示词。
5. SDP、SDPG、NER 当前只用于第一、二阶段观察，不进入正式抽取链路。

## 4. 共享配置

### 4.1 LTP 模型

第1至第4阶段默认使用：

```text
LTP/base
```

运行环境：

```bash
conda activate minimind
```

### 4.2 最小分词词典

项目根目录：

```text
segmentation_words.txt
```

词典当前保留 15 个容易被错误切分、且切分后可能改变业务含义或干扰句法
分析的最小词。默认注册词频：

```text
freq=2
```

词典只保护 CWS 分词边界，不直接决定 POS、DEP、SRL 或最终定语关系。
普通英文缩写和 LTP 已能稳定处理的组合词不进入词典，以降低对后续任务的
连锁干扰。

## 5. 第1阶段：无词典 LTP 七任务基线

目录：

```text
1.extract_ltp_all_tasks/
```

任务：

```text
cws + pos + ner + srl + dep + sdp + sdpg
```

目的：

- 查看 LTP 对原始问句的完整输出；
- 建立无自定义词典的对照基线；
- 分析分词变化对 POS、DEP、SRL 等下游任务的影响。

运行：

```bash
conda run -n minimind python \
  "1.extract_ltp_all_tasks/extract_ltp_all_tasks.py" \
  "data/original_question.md" \
  "1.extract_ltp_all_tasks/original_question_ltp_all_tasks.md"
```

本阶段不抽取最终业务定语。

## 6. 第2阶段：自定义词典 LTP 七任务

目录：

```text
2.extract_ltp_all_tasks_自定义词典/
```

在第1阶段基础上加载 `segmentation_words.txt`，再次运行全部七项任务。

运行：

```bash
conda run -n minimind python \
  "2.extract_ltp_all_tasks_自定义词典/extract_ltp_all_tasks_custom_dict.py" \
  "data/original_question.md" \
  "2.extract_ltp_all_tasks_自定义词典/original_question_ltp_all_tasks_custom_dict.md"
```

本阶段的目标不是“词越多越好”，而是通过与第1阶段对比，保留最小且必要的
分词干预。例如，词典改变 token 边界后，SRL 可能不再识别原有谓词，因此
每次修改词典后都需要重新观察 POS、DEP 和 SRL。

## 7. 第3阶段：DEP 原始句法事实

目录：

```text
3.extract_dep_raw/
```

任务：

```text
cws + dep
```

处理逻辑：

1. 使用共享分词词典；
2. 保存每个 token 的完整 DEP 依存边；
3. 原样摘出所有 DEP 标签为 `ATT` 的直接关系；
4. 不运行 POS，不筛选、不纠错、不补边；
5. 输出可审计、不可逆信息尚未被删除的 DEP 事实基线。

本阶段刻意不做：

- POS 和词面筛选；
- 完整定语重建；
- 中心词提升；
- 关系链合并；
- 维度过滤；
- LLM 业务判断。

运行：

```bash
conda run -n minimind python \
  "3.extract_dep_raw/extract_dep_raw.py" \
  "data/original_question.md" \
  "3.extract_dep_raw/original_question_dep_raw.md"
```

当前全量结果：

```text
问题：389 条
DEP 原始 ATT：1808 条
```

原始 DEP-ATT 包含时间、数量、疑问词和错误挂接，不应直接作为 RAG
最终元数据。

## 8. 第4阶段：原子 ATT 筛选、纠错与补召回

目录：

```text
4.extract_atomic_modifier_relations/
```

任务：

```text
cws + pos + dep + sdp + srl
```

职责：

- POS 和词面规则过滤时间、数量、疑问、指示及结构词噪声；
- DEP 方向和局部路径识别量词误挂、反向 ATT 与紧凑名词短语漏边；
- SDP 的受约束 `FEAT/dFEAT/mNEG/MANN` 边补充语义特征和否定极性；
- 相邻词序恢复“负增长、及时恢复”等词法修饰；
- 并列结构传播共享中心词；
- “的”结构辅助寻找右侧名词中心词；
- SRL 只提供谓词和目标论元证据，辅助纠正过短中心词；
- 所有最终原子关系仍由原句中的两个现有 token 构成。

示例：

```text
DEP：高风险 -[ADV]→ 交付 -[ATT]→ 项目
POS：高风险/a，交付/v，项目/n
修复：高风险 → 项目
```

修复候选按证据分为：

- `high`：直接 DEP-ATT 或多个确定性结构信号共同支持；
- `medium`：紧凑名词短语等局部结构支持，供后续语义阶段裁决；
- `low`：证据不足，不自动进入第四阶段最终原子 ATT。

SRL 连续片段及旧版“接受的动词定语”列仅保留作审计和第五阶段辅助证据，
不会替换第四阶段原子 ATT。

当前全量结果：

```text
问题：389 条
DEP 原始 ATT：1808 条
POS 筛选后：1257 条
结构修复候选：182 条
第四阶段最终原子关系：1433 条
```

运行：

```bash
conda run -n minimind python \
  "4.extract_atomic_modifier_relations/extract_atomic_modifier_relations.py" \
  "data/original_question.md" \
  "4.extract_atomic_modifier_relations/original_question_atomic_modifier_relations.md"
```

旁路 SRL 证据统计：

```text
动词 ATT 候选：366 条
SRL 连续候选：72 条
旧规则接受完整片段：55 条
```

第四阶段仍然不做业务价值判断、维度过滤和最终关系合并。

## 9. 第5阶段：LLM 业务语义裁决

目录：

```text
5.extract_attributives_llm/
```

模型：

```text
qwen-plus
```

当前提示词版本：

```text
stage5-v9
```

### 9.1 输入

每条问题同时输入：

- 完整原句；
- 维度抽取后的句子与禁止字符区间；
- 第3阶段原子 ATT，编号为 `A1、A2...`；
- 第4阶段 SRL 连续候选，编号为 `S1、S2...`；
- 第4阶段 DEP 动词 ATT，编号为 `V1、V2...`。

`A/S/V` 都是证据，不是标准答案。LLM 可以接受、拒绝、修剪和合并候选，
也可以直接从原句补充候选未覆盖的关系。

### 9.2 LLM 职责

1. 识别被查询、计数、列举或判断的业务实体；
2. 只保留能改善 RAG 检索的完整定语；
3. 合并低价值原子边；
4. 恢复原因、状态、关联、动作等动词性定语；
5. 处理后置条件；
6. 对证据不足的句子主动放弃。

### 9.3 程序硬校验

模型返回后，程序继续验证：

- 批次 ID 完整且不重复；
- JSON 符合 Pydantic Schema；
- 定语和中心词均为原句连续片段；
- 定语和中心词不能重叠；
- 中心词必须等于模型识别的一个业务实体；
- 关系不能跨越标点；
- 关系不能与排除维度区间重叠；
- 时间、数量、疑问、指示和空泛谓词不能作为定语；
- 不存在的证据 ID 会被移除并记录。

单条关系校验失败不会阻塞同批次其他句子。批次结构错误会自动重试；批次
持续失败时降级为逐句请求。每个成功结果都会写入 JSONL 缓存，支持断点
续跑。

### 9.4 环境

`.env`：

```text
DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

安装依赖：

```bash
conda run -n minimind pip install -r requirements-llm.txt
```

`.env` 和 `.cache` 均禁止提交到 Git。

### 9.5 运行

只检查输入对齐，不调用 API：

```bash
conda run -n minimind python \
  "5.extract_attributives_llm/extract_attributives_llm.py" \
  "data/original_question.md" \
  "5.extract_attributives_llm/original_question_attributives_llm.md" \
  --dry-run
```

运行固定验证集：

```bash
conda run -n minimind python \
  "5.extract_attributives_llm/extract_attributives_llm.py" \
  "data/original_question.md" \
  "5.extract_attributives_llm/validation_10.md" \
  --line-ids 42,50,84,118,150,214,309,336,366,388
```

全量运行：

```bash
conda run --no-capture-output -n minimind python \
  "5.extract_attributives_llm/extract_attributives_llm.py" \
  "data/original_question.md" \
  "5.extract_attributives_llm/original_question_attributives_llm.md" \
  --workers 2
```

默认批大小为 4、温度为 0。缓存指纹包含提示词版本、模型、原句、维度区间
和全部候选证据；输入或提示词变化后不会误用旧缓存。

## 10. 当前全量结果

输入情况：

```text
有效问题：389 条
维度结果可用：360 条
维度结果不可用：29 条
排除维度区间：359 个
```

第五阶段结果：

```text
完成度：389/389
包含业务定语的问题：138 条
无业务定语的问题：251 条
最终业务定语关系：144 条
```

关系类型：

```text
cause：44
state：39
association：33
classification：16
possessive：7
action：5
```

置信度：

```text
high：136
medium：8
```

本次全量 API 用量：

```text
输入：251552 tokens
输出：46797 tokens
```

正式输出：

```text
5.extract_attributives_llm/original_question_attributives_llm.md
```

## 11. 典型数据流

原句：

```text
全球运营商业务产品质量导致的网上事故有多少个
```

维度禁止区间：

```text
全球运营商
```

第三阶段：

```text
业务 → 产品
产品 → 质量
导致 → 事故
网上 → 事故
```

第四阶段：

```text
全球 → 运营商
运营商 → 业务
业务 → 产品
产品 → 质量
导致 → 事故
网上 → 事故
```

第四阶段同时保留但不用于替换原子 ATT 的 SRL 证据：

```text
全球运营商业务产品质量导致的 → 事故
```

第五阶段去除查询作用域并提升中心词：

```text
产品质量导致的 → 网上事故
```

最终文本仍全部逐字来自原句。

## 12. 已知边界

### 12.1 业务实体边界仍会导致漏抽

当前要求 `head_text` 必须完全等于某个业务实体。这能阻止过短中心词，但
模型把“定语 + 实体”整体识别为实体时，会错误过滤语义正确的关系。

当前已观察到：

```text
未关闭的 → 管理升级单
重大或高风险 → 交付EI项目
中风险 → 交付项目
```

需要改进实体识别和关系切分，而不是简单删除重叠校验。

### 12.2 维度结果不可用时约束较弱

维度文件空行或末尾缺失时，程序按“不可用”处理，不会把整句误当成已删除
维度。但这也意味着只能依靠提示词排除时间和查询范围。

当前仍存在把带时间或范围的长片段误判为定语的风险，例如需要重点复核
原文件第 390 行。

### 12.3 LTP 候选受分词影响

自定义词典改变 token 边界后，POS、DEP、SRL 都可能变化。词典只能保留
最小必要词，不适合扩展成完整业务术语表。

### 12.4 还没有人工金标准

当前质量主要依靠程序约束和典型样例检查。144 条关系并不等于已经达到可
上线精度；在写入 RAG 索引前仍需要人工标注评测。

## 13. 下一步建议

优先级建议：

1. 建立 50–100 条人工金标准，覆盖普通 ATT、动词性定语、后置条件、
   多层实体、维度删除和无有效定语；
2. 将“业务实体”进一步拆分为核心实体、完整实体短语和指标，避免
   `国家/国家数量`、`项目/交付EI项目`混为一层；
3. 修正实体边界造成的三类已知漏抽，并为硬校验添加单元测试；
4. 为维度结果不可用的情况增加确定性时间/查询范围校验；
5. 计算关系级 Precision、Recall、F1、整句完全匹配率和维度泄漏率；
6. 评测稳定后，再将最终关系写入 RAG 的结构化元数据。

## 14. 设计结论

当前路线的核心不是让某一个模型完成全部任务，而是分层承担风险：

```text
最小词典保证分词边界
        +
POS/DEP提供可解释的高召回原子关系
        +
SRL恢复动词论元和连续片段
        +
LLM判断业务价值、合并和补漏
        +
Schema与字符区间规则保证输出边界
```

这种设计比直接使用 DEP 原子边更符合业务 RAG 需求，也比完全依赖 LLM
更容易审计、定位错误和重复运行。
