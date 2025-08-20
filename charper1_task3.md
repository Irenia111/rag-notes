# 四步构建RAG 笔记

## 核心代码
```Python
# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)


# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# rag 的结果会作为 prompt 的 context
answer = llm.invoke(prompt.format(question=question, context=docs_content))

# 只打印 content，没有元数据
print(answer.content)
```

## 默认`chunk` 参数

```python
text_splitter = RecursiveCharacterTextSplitter()
```
使用默认 chunk_size 和 chunk_overlap，通常是 **1000 字符** 的上限（具体依赖版本），且 **重叠为 0**。

根据 LangChain 官方文档说明，RecursiveCharacterTextSplitter 的行为机制如下：
•	它尝试按分隔符列表依次拆分文本，默认列表为 ["\n\n", "\n", " ", ""]，即先按段落，再按换行符，再按空格，若都不行才进行字符拆分  ￼ ￼。
•	chunk_size 是通过 length_function 衡量的最大字符数。例如 len() 会以字符数为准  ￼。
•	如果某个拆分后的文本块长度仍超过 chunk_size，Splitter 会尝试使用下一个分隔符进一步拆分  ￼。

而在 Stack Overflow 上，也有类似说明，当指定的 chunk 超出 size，但没有可用 separator 时，仍可能返回过大的 chunk  ￼。

## 修改 `chunk` 参数
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,            # 每个块最多500个字符
    chunk_overlap=50,          # 块与块之间重叠50个字符（可选）
    length_function=len,       # 用字符长度衡量块大小
    # separators 参数可以自定义分隔符顺序
)
```

修改后的返回
```text
文中举了以下例子：

1. **强化学习例子**：
   - DeepMind研发的走路智能体：智能体学习在曲折道路上行走，通过举手保持平衡以更快前进，并能适应环境扰动。
   - 机械臂抓取：通过强化学习训练机械臂抓取不同形状的物体，避免传统算法需单独建模的耗时问题。

2. **探索与利用的例子**：
   - 选择餐馆：利用是去已知喜欢的餐馆，探索是尝试新餐馆。
   - 做广告：利用是采用最优广告策略，探索是尝试新策略。
   - 挖油：利用是在已知地点挖油，探索是在新地点尝试。
   - 玩游戏（如《街头霸王》）：利用是重复固定策略（如蹲角落出脚），探索是尝试新招式（如“大招”）。

3. **奖励的例子**：
   - 象棋选手：最终奖励为正（赢）或负（输）。
```

### 修改 `chunk` 参数后的差异

1. 主题与覆盖面

- **配置前**：列举了更广泛的例子（羚羊、股票、Atari、餐馆、广告、挖油、街头霸王），更像是整章里“探索/利用”概念的**通篇扫读**。

- **配置后**：内容被**主题化重组**（“强化学习例子 / 探索与利用 / 奖励”），同时引入了**新的具体案例**（DeepMind 走路智能体、机械臂抓取、象棋），而丢失了“羚羊、股票、Atari”等原例子。

  这很典型：改变 chunk_size/overlap 会让检索命中的**文本片段换了一批**，从而影响模型看到的证据与回答组织方式。

2. 结构与表达

- **配置前**：一串平铺直叙的要点列表。

- **配置后**：按主题分组、二级缩进更清晰，**摘要感更强**。

  较小或更合理的 chunk 往往能把“同一小节”的语义捆在一起，利于生成更有结构的答案；相反，过大的 chunk 可能混入不相干内容，过小的则容易割裂上下文。

3. 长度与成本（来自你贴的元数据）

- Prompt tokens：**5549 → 796（↓约85.7%）**

- Total tokens：**5754 → 1027（↓约82.2%）**

- Completion tokens：**205 → 231（↑约12.7%）**

  说明：更紧凑的切分 + 更少检索命中，**显著降低了提示词开销与总成本**；回答略长，是因为模型在更聚焦的证据上做了“主题化整理”。这与业界经验一致：合适的 chunking 可**降低延迟/成本**，同时提升相关性或可读性。


**为什么会这样（机制）**
- RecursiveCharacterTextSplitter 会按分隔符层级（段落 → 行 → 空格 → 字符）递归切分，并以 chunk_size 为上限、chunk_overlap 做重叠；不同参数组合会改变“每个向量”所代表的语义范围，从而改变相似度检索的命中集合。
- 检索阶段（如 InMemoryVectorStore.similarity_search(k=...) 或 MMR）会基于嵌入向量选回前 k 个块；当块更小/更聚焦时，常出现**命中文本发生迁移**，导致内容取样和侧重点变化。


## LlamaIndex代码注释

```python
import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 配置大语言模型
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))

# 配置嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# ---（可选）进一步的文档切分/索引构建细化设置 ---
# 默认情况下，LlamaIndex 会使用内置的 node_parser 将文档切分成节点再向量化。
# 如果需要自定义“分块大小/重叠”等，可启用如下配置（示例）：
#
# from llama_index.core.node_parser import SentenceSplitter
# Settings.node_parser = SentenceSplitter(
#     chunk_size=500,      # 每个块最大字符数（控制检索粒度）
#     chunk_overlap=50     # 相邻块重叠字符数（保持上下文连续）
# )
#
# 也可以按 Markdown 标题分块、按句子/段落分块，或用语义切分器，视任务场景选择。

# 读取本地文件
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 构建向量存储
#    from_documents 会：
#       a) 将 docs 切分为“节点”（可选受 Settings.node_parser 影响）
#       b) 用 Settings.embed_model 计算每个节点的向量
#       c) 将向量存入默认的向量存储（内存型，适合小规模/快速实验）
index = VectorStoreIndex.from_documents(docs)


# 获取查询引擎（Query Engine）
# Query Engine 会把“用户问题”向量化，与索引中的文档向量做相似度检索，
# 取回最相关的节点作为上下文，交给 LLM（Settings.llm）生成最终答案（RAG 流程）
query_engine = index.as_query_engine()

# 打印该 Query Engine 内部用到的 Prompt 模板
print(query_engine.get_prompts())


# 发起 query
print(query_engine.query("文中举了哪些例子?"))
```