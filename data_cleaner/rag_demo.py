import chromadb
from sentence_transformers import SentenceTransformer


# ── 1. 准备文档（你的学习笔记）──────────────────────
documents = [
    "RAG是检索增强生成技术,先检索相关文档再让大模型回答，解决幻觉问题",
    "Embedding是把文字变成数字向量,语义相近的词向量方向接近",
    "ChromaDB是向量数据库,用来存储和检索Embedding向量",
    "json.loads()把字符串转成字典,json.dumps()把字典转成字符串",
    "temperature=0保证大模型每次输出格式一致,合适结构输出场景",
    "safe_json_parse()是防御性解析函数,处理模型输出的json代码包裹",
    "drop_duplicates()删除重复行,str.strip()去掉首尾空格,str.lower()转小写",
    "try/except捕获程序异常,主动防御用fi/else提前判断,被动兜底用except",
    "TotalCharges字段含空格伪装的空值,isnull()无法直接检测需要先strip",
    "Self-Attention四步: Q乘以K转置,除以根号dk,softmax,乘以V",


]

doc_ids = [f"doc_{i}" for i in range(len(documents))]

# ── 2. 加载 Embedding 模型 ────────────────────────────
print("加载Embedding模型...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("模型加载完成")

# ── 3. 离线阶段：建库 ─────────────────────────────────
print("\n离线阶段:建库中...")
client = chromadb.Client()
collection = client.create_collection("study_notes")

embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=doc_ids

)
print(f"建库完成,共存入{len(documents)}条笔记")

# ── 4. 在线阶段：查询 ─────────────────────────────────
questions = [
    "json怎么解析",
    "怎么处理大模型输出不稳定的问题",
    "数据清洗怎么处理空值",

]

print("\n在线阶段: 检索中...")
for q in questions:
    q_embedding = model.encode([q]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=2
    )

    print(f"\n问题: {q}")
    print(f"最相关的笔记:")
    for doc in results["documents"][0]:
        print(f" -> {doc}")