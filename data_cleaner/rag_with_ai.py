import chromadb
from sentence_transformers import SentenceTransformer
from safe_json_parse import safe_json_parse
from openai import OpenAI
import os
import getpass

# ── 1. 初始化 ──────────────────────────────────────
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    api_key = getpass.getpass("请输入阿里云 DashScope API Key: ")
    os.environ["DASHSCOPE_API_KEY"] = api_key

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ── 2. 建库（离线阶段）────────────────────────────
documents = [
    "RAG是检索增强生成技术，先检索相关文档再让大模型回答，解决幻觉问题",
    "Embedding是把文字变成数字向量，语义相近的词向量方向接近",
    "ChromaDB是向量数据库，用来存储和检索Embedding向量",
    "json.loads()把字符串转成字典，json.dumps()把字典转成字符串",
    "temperature=0保证大模型每次输出格式一致，适合结构化输出场景",
    "safe_json_parse()是防御性解析函数，处理模型输出的json代码块包裹",
    "drop_duplicates()删除重复行，str.strip()去掉首尾空格，str.lower()转小写",
    "try/except捕获程序异常，主动防御用if/else提前判断，被动兜底用except",
    "TotalCharges字段含空格伪装的空值，isnull()无法直接检测需要先strip",
    "Self-Attention四步：Q乘以K转置，除以根号dk，Softmax，乘以V",
]

db = chromadb.Client()
collection = db.create_collection("notes")
embeddings = model.encode(documents).tolist()
collection.add(
    documents = documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]

)
print(f"建库完成，共存入{len(documents)}条笔记\n")


# ── 3. RAG 查询函数（在线阶段）───────────────────
def rag_query(question: str) -> str:
    # 第一步：检索相关笔记
    q_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=2)
    context = "\n".join(results["documents"][0])


    # 第二步：把笔记 + 问题组装成 prompt
    response = client.chat.completions.create(
        model="qwen-turbo",
        temperature=0,
         messages=[
            {
                "role": "system",
                "content": "你是学习助手。只根据提供的参考资料回答问题，不要用自己的知识补充。"
            },
            {
                "role": "user",
                "content": f"参考资料：\n{context}\n\n问题：{question}"
            }
        ]
    )
    return response.choices[0].message.content

# ── 4. 测试三个问题 ────────────────────────────────
questions = [
    "json怎么解析?",
    "怎么处理大模型输出不稳定的问题？",
    "数据清洗怎么处理空值？",

]

for q in questions:
    print(f"问题:{q}")
    answer = rag_query(q)
    print(f"回答:{answer}")
    print("="* 50)
    