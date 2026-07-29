# 3. DEP 原始句法事实

第三阶段是可追溯的 DEP 事实层：

```text
cws（分词前置）+ dep
```

目录名中的 `pos_dep` 是历史名称；当前代码不运行 POS，也不使用 POS
筛选关系。

## 本阶段做什么

1. 使用项目根目录的 `segmentation_words.txt` 干预 LTP 分词。
2. 运行 `LTP/base` 的 `cws + dep`。
3. 保存每个 token 的完整依存边，包括中心词位置和依存标签。
4. 原样摘出所有 DEP 标签为 `ATT` 的边。

第三阶段回答的是“DEP 模型输出了什么”，不判断这些关系是否应当用于
业务 RAG。这样即使 POS 或后续规则判断错误，原始依存证据也不会丢失。

## 本阶段不做什么

- 不运行 POS，不根据词性删除关系。
- 不过滤时间、数量、疑问、指示或结构词。
- 不修正 DEP 中心词。
- 不在局部窗口补充漏掉的关系。
- 不重建完整定语。
- 不过滤维度，不合并关系。
- 不使用 SRL、SDP、LLM、Schema 或语义改写。

上述筛选、纠错和补召回全部放在第四阶段。

## 当前配置

- 模型：`LTP/base`
- 共享词典：项目根目录 `segmentation_words.txt`
- 自定义词频：`freq=2`
- 输入：`data/original_question.md`
- 输出：`original_question_dep_raw.md`

## 运行命令

```bash
conda run --no-capture-output -n minimind python \
  "3.extract_dep_raw/extract_dep_raw.py" \
  "data/original_question.md" \
  "3.extract_dep_raw/original_question_dep_raw.md"
```

## 输出列

- 原文件行号
- 原句
- CWS 分词
- 完整 DEP
- DEP 原始 ATT
