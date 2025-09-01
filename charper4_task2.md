# 查询构建 笔记

## 代码
```python
import os
from langchain_deepseek import ChatDeepSeek 
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
try:
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    for doc in docs:
        original = doc.metadata
        
        # 提取基本元数据字段
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        doc.metadata = metadata
        bili.append(doc)
        
except Exception as e:
    print(f"加载BiliBili视频失败: {str(e)}")

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数）",
        type="integer"
    )
]

# 4. 创建自查询检索器
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
    )

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

# 5. 执行查询示例
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    results = retriever.invoke(query)
    if results:
        for doc in results:
            title = doc.metadata.get('title', '未知标题')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("="*50)
    else:
        print("未找到匹配的视频")
```
1. 加载视频数据

使用 BiliBiliLoader 从指定 URL 抓取视频信息。

提取并整理出几个关键元数据字段：
```text
title（标题）

author（作者）

source（视频 ID）

view_count（观看次数）

length（视频时长，秒）
```
存放到 bili 列表，供后续使用。

2. 创建向量存储

使用 BAAI/bge-small-zh-v1.5 中文嵌入模型。

将视频文档和元数据存入 Chroma 向量数据库。
👉 后续查询会基于向量相似度 + 元数据条件。

3. 定义元数据字段信息

通过 AttributeInfo 声明字段：

- 名称、说明、类型。

这一步相当于告诉检索器哪些字段可供过滤（如 “长度 > 600”）。

4. 创建自查询检索器

使用 ChatDeepSeek 作为大模型。

SelfQueryRetriever 可以把用户的自然语言问题，转化为：

- 向量搜索（基于内容相似度）

- 元数据过滤（基于字段条件，比如 “时长大于600秒”）。

- enable_limit=True：允许结果数量限制。

5. 执行示例查询

示例问题：

- “时间最短的视频”

- “时长大于600秒的视频”

运行后会返回满足条件的视频，并输出其标题、作者、观看次数和时长。


## 为何代码中查询“时间最短的视频”时，得到的结果是错误的

运行代码进行查询
```text

--- 查询: '时间最短的视频' ---
INFO:httpx:HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:langchain.retrievers.self_query.base:Generated Query: query=' ' filter=None limit=1
标题: 《吴恩达 x OpenAI Prompt课程》【专业翻译，配套代码笔记】02.Prompt 的构建原则
作者: 二次元的Datawhale
观看次数: 18788
时长: 1063秒
==================================================

--- 查询: '时长大于600秒的视频' ---
INFO:httpx:HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:langchain.retrievers.self_query.base:Generated Query: query=' ' filter=Comparison(comparator=<Comparator.GT: 'gt'>, attribute='length', value=600) limit=None
WARNING:chromadb.segment.impl.vector.local_hnsw:Number of requested results 4 is greater than number of elements in index 3, updating n_results = 3
标题: 《吴恩达 x OpenAI Prompt课程》【专业翻译，配套代码笔记】03.Prompt如何迭代优化
作者: 二次元的Datawhale
观看次数: 7090
时长: 806秒
==================================================
标题: 《吴恩达 x OpenAI Prompt课程》【专业翻译，配套代码笔记】02.Prompt 的构建原则
作者: 二次元的Datawhale
观看次数: 18788
时长: 1063秒

```


这次的“错误”其实是**预期行为**：`SelfQueryRetriever` 只会把自然语言转成「文本相似度检索 + 元数据过滤」，**不支持排序/聚合**。
“时间最短的视频”属于 **超级lative（最短/最长/最大/最小）** 问题，需要“按 `length` 升序排序后取前 1 条”。从日志看：

```
Generated Query: query=' ' filter=None limit=1
```

* `query=' '`：模型认为没有可用的文本查询（确实，“最短”不是文本语义条件），于是给了空查询。
* `filter=None`：没有可翻译成过滤条件（因为是排序需求，不是范围条件）。
* `limit=1`：只限制数量为 1，但**没有排序**，所以返回的是“任意相似度第一”的文档，而不是“时长最短”的文档。

反而第二个查询“时长大于600秒的视频”能正常工作，是因为它能翻译成**范围过滤**：

```
filter=Comparison(GT, attribute='length', value=600)
```

---

* 像 “最短/最长/前N个/TopK/最大观看数” 这类请求，**都需要排序或聚合**，默认的 `SelfQueryRetriever` 都处理不了，需要在上层加一层**后处理**。
* 对于范围条件（`>、>=、<、<=`），`SelfQueryRetriever` 可以胜任，比如看到的“>600 秒”就很好用。
