# 5. LLM业务定语筛选与合并

第五阶段不替代 LTP，而是在第3、4阶段的候选证据上完成业务语义裁决：

```text
原始问题
+ 维度禁止区间
+ POS/DEP原子ATT
+ SRL连续定语候选
        ↓
qwen-plus语义筛选、合并和遗漏补充
        ↓
原文跨度与维度重叠校验
```

## 职责

- 原句是唯一事实来源。
- 第3、4阶段结果只是可错、可缺失的候选证据。
- LLM可以保留、拒绝、修剪、合并候选，也可以补充后置定语。
- 只输出能缩小RAG检索范围的完整业务定语。
- 定语和中心词必须逐字连续出现在原句中，不允许改写。
- 已抽取维度转换成原句字符禁止区间，最终关系不得与其重叠。
- 维度文件中的空行和缺失行按“维度结果不可用”处理，不会把整句误删。
- 不使用SDP或SDPG。

## 确定性校验

程序会丢弃并记录：

- 不在原句中的定语或中心词；
- 与已抽取维度区间重叠的关系；
- 跨越句内标点的片段；
- 纯时间、数量、疑问和指示修饰；
- 单独的“导致、相关、涉及、有、是”等空泛谓词；
- 引用不存在候选ID时会删除非法ID并记录；关系本身仍须通过原文、
  维度和实体校验。

JSON结构、批次ID等结构错误会反馈给模型并重试；单条关系违反
原文跨度或维度硬约束时会被确定性丢弃，不会阻塞同批次其他问题。
批次持续失败时自动降级为逐句请求；单句仍失败则记录失败原因并继续。

如果维度区间恰好完整覆盖定语的前缀或后缀，程序会把该边界投影掉；
如果维度位于定语中间、删除后会形成非连续文本，则仍然丢弃该关系。
当前还会在原因、状态、关联和动作关系中，裁掉紧邻已排除维度之后的
稳定查询作用域前缀“业务”；该规则不参与分词，也不扩展为业务词典。

## 环境

使用项目根目录 `.env`：

```text
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

依赖见项目根目录的 `requirements-llm.txt`。

## 先检查输入

```bash
conda run -n minimind python \
  "5.extract_attributives_llm/extract_attributives_llm.py" \
  "data/original_question.md" \
  "5.extract_attributives_llm/original_question_attributives_llm.md" \
  --dry-run
```

## 指定样本验证

```bash
conda run -n minimind python \
  "5.extract_attributives_llm/extract_attributives_llm.py" \
  "data/original_question.md" \
  "5.extract_attributives_llm/validation_10.md" \
  --line-ids 42,50,84,118,150,214,309,336,366,388
```

## 全量运行

```bash
conda run -n minimind python \
  "5.extract_attributives_llm/extract_attributives_llm.py" \
  "data/original_question.md" \
  "5.extract_attributives_llm/original_question_attributives_llm.md"
```

默认设置：

- 模型：`qwen-plus`
- 批大小：4
- 并发：1
- 缓存：输出目录下的 `.cache/*.jsonl`
- 温度：0
- 输入指纹包含提示词版本、原句、维度区间及所有候选证据

可以使用 `--workers 2` 提高并发；建议先在盲测样本上确认质量，再运行全量数据。
