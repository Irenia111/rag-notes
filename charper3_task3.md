# 本地向量存储 笔记

## 测试代码

```Python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")` 
```


输出结果：
```text
LlamaIndex 索引已保存至: ./llamaindex_index_store
```
代码主要完成了三个关键任务：配置嵌入模型，创建文档索引，以及将索引持久化保存。

1. 配置全局嵌入模型
```python
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
```
- Settings 是 LlamaIndex 提供的一种配置工具，用于在整个索引与查询流程中统一管理嵌入模型、LLM、chunk 大小等资源。
- HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5") 指定使用该模型来生成文本嵌入。此嵌入模型会被用于将文档（以及后续的查询）转换为向量，确保在创建和检索索引时使用相同的 embedding 模型。

2. 构建文档列表
```python
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]
```
- 通过定义一组文本，调用 Document(text=...) 模型将这些文本转化为 Document 对象。
- Document 是 LlamaIndex 用于承载文本及其相关元数据（metadata）的基本单元

3. 创建向量索引并持久化
```python
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")
```
- 使用 VectorStoreIndex.from_documents(docs) 构建一个向量索引：
    - 文档被分成更小的 chunks（如果需要），并转换为 embeddings。 
    - 这些 embeddings、原始文本 chunk 以及相应的文档结构被组织成索引结构（例如节点、向量存储、索引 metadata），以便后续进行快速相似性检索。
- index.storage_context.persist(persist_dir=persist_path) 将整个索引存盘： 
  - 默认情况下，包括文档 store、索引 metadata store、向量 store（embeddings）都会被保存为文件（如 .json 格式），存放在指定目录下。 
  - 这样做的好处是，下次可以直接从磁盘加载，无需重新处理文档或生成 embeddings，从而节省时间和资源。

## 实现对LlamaIndex存储数据的加载和相似性搜索

```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simple_llamaindex_query.py

一个极简版本的 LlamaIndex 测试脚本：
- 嵌入模型、持久化目录、查询文本和 top_k 均已硬编码。
- 加载向量索引后执行相似性检索，并打印结果。
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# —— 1. 全局变量：硬编码路径和参数 —— #
PERSIST_DIR = Path("./llamaindex_index_store")  # 保存索引的目录
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"         # 嵌入模型，必须与构建索引时所使用的嵌入模型一致
QUERY_TEXT = "LlamaIndex 是什么？"               # 测试查询
TOP_K = 3                                      # 返回相似结果数目

# —— 2. 嵌入模型配置 —— #
Settings.embed_model = HuggingFaceEmbedding(EMBED_MODEL)

# —— 3. 加载索引 —— #
if not PERSIST_DIR.exists() or not PERSIST_DIR.is_dir():
    raise FileNotFoundError(f"未找到索引目录：{PERSIST_DIR}。请先执行持久化步骤。")

storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
index = load_index_from_storage(storage_context)

# —— 4. 执行相似性搜索 —— #
# 最基础的向量检索（vector-based retrieval），常称为“top‑k 嵌入相似度检索”，它的检索逻辑只涉及向量间的距离计算（如余弦相似度）。这是一种高效且常见的检索方式。
retriever = index.as_retriever(similarity_top_k=TOP_K)
results = retriever.retrieve(QUERY_TEXT)

# —— 5. 输出展示结果 —— #
print("\n=== 相似性搜索结果 ===")
print(f"查询：{QUERY_TEXT}\n")
if not results:
    print("无匹配结果。")
else:
    for idx, r in enumerate(results, start=1):
        score = r.score
        text = r.get_text()
        meta = r.metadata or {}
        header = f"#{idx}  score={score:.4f}"
        src = meta.get("file_name") or meta.get("source") or meta.get("doc_id")
        if src:
            header += f"  source={src}"
        print(header)
        print("-" * len(header))
        snippet = " ".join(text.replace("\n", " ").split())
        print(snippet[:400] + ("..." if len(snippet) > 400 else ""))
        if meta:
            filtered = {k: v for k, v in meta.items() if k not in {"text", "chunk"}}
            if filtered:
                print("[metadata]", filtered)
        print()
```


输出结果
```text

=== 相似性搜索结果 ===
查询：LlamaIndex 是什么？

#1  score=0.6577
----------------
LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。

#2  score=0.3267
----------------
它提供了数据连接、索引和查询接口等工具。

#3  score=0.2273
----------------
张三是法外狂徒
```

### 关于 embed 模型
EMBED_MODEL 必须与构建索引时所使用的嵌入模型一致
为什么两者必须相同？
- 一致性重要：使用的嵌入模型决定了文本转换为向量的方式。若建库时用的是 A 模型，将查询向量用 B 模型生成（不一致），两者在向量空间上就不可比，导致检索结果失效或报错。(docs.llamaindex.ai)  ￼
- 维度不匹配问题：这就是实际场景中常见的“维度对不上”的错误——比如索引存储的是 768 维的向量，可却用 1536 维的模型来生成查询嵌入，导致计算无法进行。(github.com issue)  ￼
- 官方建议：文档中也特别强调确保“用于索引和查询的嵌入模型必须一致”。(docs.llamaindex.ai 基本策略)  ￼
- 不同模型不可混用：在多个索引使用不同嵌入模型的场景下，推荐在元数据中存储所用模型的信息，并在检索时匹配相应的模型来生成查询向量，确保一致性。(DeepLearning.AI 社区讨论)

构建索引时使用 A 模型，查询时使用 B 模型  -> 向量空间不匹配，检索失效或报错
使用不同模型且忽略模型信息 -> 检索质量不可控，会严重影响结果
正确匹配模型  -> 向量比较有效，检索准确

### LlamaIndex 支持更多检索模式
1. 不同的检索模式（retrieval_mode）
如文档中提到，as_retriever() 支持不同模式，特别是针对特定索引（如 SummaryIndex）时，可以通过 retriever_mode 参数切换行为。例如：
```python
retriever = summary_index.as_retriever(
    retriever_mode="llm",
    choice_batch_size=5,
)
```
这通常用于摘要索引或更复杂场景的查询方式。 ￼

此外，还有一些特殊的检索模式：
- files_via_metadata：基于文件元数据检索整体文件片段，如用户直接提问某文件内容。 
- files_via_content：基于内容检索文件。 
- auto_routed：利用轻量 Agent（策略）自动判断使用哪个检索模式最合适。

2. BM25 检索
这是基于传统信息检索算法（TF‑IDF 的改进）的检索方式，可以用于补充或替代向量检索。可用 BM25Retriever 来创建这样的检索器：
```python
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=5
)
```
该检索器适合关键词召回，不依赖嵌入向量。

3. 递归检索（Recursive Retrievers）

这种策略允许链式、多层次检索：
- 首先检索概要或相关节点（summary），
- 然后“深入”获取更详细的文档片段。
适用于文档结构复杂或信息分散的场景，能提高检索的相关性和内容完整性。 ￼

4. 两阶段检索（Embedding + LLM Re-Ranking）

这是一个混合策略：
- 第一阶段：使用嵌入检索获取一批候选文档（Top-K，通常 K 较大以提高召回）。
- 第二阶段：用 LLM 对候选文档进行基于语义的精确排序（re-ranking），以提升最终结果质量。

### 关于相似度
相似度分数（score）表示什么？简单来说，它衡量了查询文本的嵌入向量与文档（或节点）嵌入向量之间的“接近程度”。在 LlamaIndex 中，该分数越高，代表语义相似度越强，也就是说内容更相关。
- score 表示查询与文档在向量空间的相似度（通常为 cosine similarity）
- 通常接近 1 表示高度相似，接近 0 表示无相关性
- 越高意味着语义越匹配，越相关
- 代码中检索结果按 score 降序排列，返回最“靠近”的文档

---

相似度分数具体含义是什么？
- 在 LlamaIndex 的检索结果中，score 通常是 余弦相似度（cosine similarity），用来衡量两个向量之间方向的相似程度。更高的值表示向量越“指向同一方向”，意味着语义越相似。
- 从讨论来看，LlamaIndex 中的分数反映 “两个非零向量之间夹角的余弦值”，分数越接近 1 越相似。 ￼
---

向量相似度基础简介

为了更全面一点，这里补充一些背景：
- **余弦相似度（Cosine Similarity）**
计算公式为：向量 A 和 B 的点积除以它们的模长乘积
\text{cosine\_similarity} = \frac{A \cdot B}{\|A\|\|B\|}
- 值域通常是 -1, +1： 
  - +1 表示完全相同方向 
  - 0 表示完全正交（无相关性） 
  - -1 表示相反方向
- 在文本向量中，由于嵌入值多为非负，实际值区间常在 [0, 1]。 ￼ 
- 为什么用余弦相似度: 它只关注方向（即语义），不受向量长度影响，适合比较不同长度文本。 ￼


