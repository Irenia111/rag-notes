# 数据加载 笔记

## 核心代码
```Python
from unstructured.partition.auto import partition

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
```
输出结果：
```text
解析完成: 279 个元素, 7500 字符
元素类型: {'Header': 22, 'Title': 195, 'UncategorizedText': 41, 'NarrativeText': 3, 'Footer': 15, 'ListItem': 3}
```

partition函数参数解析：

- filename: 文档文件路径，支持本地文件路径
- content_type: 可选参数，指定MIME类型（如"application/pdf"），可绕过自动文件类型检测
- file: 可选参数，文件对象，与filename二选一使用
- url: 可选参数，远程文档URL，支持直接处理网络文档
- include_page_breaks: 布尔值，是否在输出中包含页面分隔符
- strategy: 处理策略，可选"auto"、"fast"、"hi_res"等
- encoding: 文本编码格式，默认自动检测
- 
partition函数使用自动文件类型检测，内部会根据文件类型路由到对应的专用函数（如PDF文件会调用partition_pdf）。

根据 Unstructured 官方文档的说明：

“If you call the partition function, unstructured will attempt to detect the file type and route it to the appropriate partitioning function.”
也就是说，默认情况下，当你调用 partition(...) 并传入 PDF 文件（如 .pdf 文件或 content_type="application/pdf"），它会自动调用专门的 partition_pdf 函数来处理该 PDF 文档。 ￼

官方特别指出：
- partition 是一个简化接口，其使用自动文件类型检测机制，根据文件后缀或内容类型判断文件类型，然后将其委派给相应的专用函数（如 PDF—partition_pdf，HTML—partition_html 等）。
- 所有通过 partition 路由调用的“专用函数”，都会使用其默认参数进行调用。 ￼

因此，如果你只是用 partition(...) 来处理 PDF，内部确实是在调用 partition_pdf，但调用的是其默认设置。

如果你对 PDF 处理有更特定的需求，比如设置 OCR 语言、启用图像提取、使用更高精度的布局分析策略，或者启用表格结构推理等，就 应该直接使用 from unstructured.partition.pdf import partition_pdf，以便传入更高级、更多样化的参数，从而获得更专业、可控且可能更高性能的处理效果。

## 使用 partition_pdf

```python
from unstructured.partition.pdf import partition_pdf 

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

elements = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only",                  # 可选：hi_res / fast / ocr_only / auto
    pdf_infer_table_structure=True,     # 推断并结构化表格
    extract_images_in_pdf=True,         # 抽取图片
    languages=["chi_sim"],  # 指定简体中文 OCR 识别
    include_page_breaks=True            # 包含分页符
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
```

### 使用 `strategy="hi_res"` 输出
```text
解析完成: 244 个元素, 8257 字符
元素类型: {'Image': 21, 'UncategorizedText': 104, 'Header': 4, 'NarrativeText': 68, 'Table': 4, 'FigureCaption': 4, 'Title': 30, 'ListItem': 4, 'PageBreak': 5}
```

###  `strategy="ocr_only"` 输出
```text
解析完成: 106 个元素, 2103 字符
元素类型: {'Title': 68, 'UncategorizedText': 27, 'NarrativeText': 6, 'PageBreak': 5}
```


## 关于 strategy

- hi_res 输出出现 'Image', 'Table', 'FigureCaption', 'Header', 'Title', 'NarrativeText', 'ListItem' 等多类别 → **版面模型在工作**，适合做表格解析、图文配对、层级摘要等下游任务。

- ocr_only 输出主要是 'Title'/'UncategorizedText'/'NarrativeText' → **更偏“拿到文字就好”** 的简单场景。

hi_res 比 ocr_only 把**更多版面元素**识别出来（表格、图片、标题、段落、页眉等），所以你看到 _244 个元素_，而 ocr_only 主要做 **纯 OCR** 的文字提取，版面分类更少，所以只有 _106 个元素_、字符数也更少。这正符合两种策略的设计差异。

**选哪一个更合适？**

- **PDF 有可抽取文本（非扫描件）且你更关心速度/纯文本**
  用 strategy="fast"（必要时遇到不可抽取文本再自动回落到 OCR）。

- **需要高质量版面理解（表格、图片、标题层级、图注等）**
  用 strategy="hi_res"（模型做版面检测 → 元素更丰富、更易下游结构化）。

- **纯扫描件或你想强制逐页 OCR**
  用 strategy="ocr_only"（元素类型更少，但对图片/扫描件稳）。



**让**  **hi_res** **更“会提取”**

1. **表格**

   已经开了 pdf_infer_table_structure=True，还可以：
  - 选择/指定更合适的版面模型：hi_res_model_name="yolox"（或 detectron2_onnx）以提升检测速度/效果；新文档推荐用 hi_res_model_name 而不是已弃用的 model_name。
  - 若发现表格漏检，在 API 侧常用 skip_infer_table_types=[] 之类参数避免跳过某些表格类型（供思路参考）。


2. **图片与图注**

   你已经启用 extract_images_in_pdf=True；这在 hi_res 下才生效，用于把图片作为独立元素/资源导出，便于做图文对齐或检索。

3. **语言（OCR）**

   对于包含中文的页面（当 hi_res 触发 OCR 或 ocr_only），加上：

```python
    languages=["chi_sim", "eng"]
```
	以提升中文/中英混排识别质量（需安装相应 Tesseract 语言包）。
