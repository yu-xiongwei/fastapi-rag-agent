from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

# ── 1. 初始化组件 ──────────────────────────────────
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key="sk-4becd4ec98e6435293b76cb8ed7fbcaf",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",

)

# ── 2. 建向量库 ────────────────────────────────────
documents = [
    "RAG是检索增强生成技术,先检索相关文档再让大模型回答，解决幻觉问题",
    "json.loads()把字符串转成字典,json.dump()把字典转成字符串",
    "temperature=0保证大模型每次输出格式一致,适合结构化输出场景",
    "TotalChagres字段含空格伪装的空值,isnull()无法直接检测需要strip()",
    "drop_duplicates()删除重复行,str.strip()去掉首尾空格,str.lower()转小写",
]

vectorstore = Chroma.from_texts(documents,embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":2 })

# ── 3. 定义 Prompt 模板 ───────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system","你是学习助手，只能根据以下参考资料回答问题,不要用自己的知识补充。\n\n参考资料:\n{context}"),
    ("user","{question}")
])

# ── 4. 组装 RAG Chain ──────────────────────────────
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

chain = (
    {"context" : retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 5. 测试 ────────────────────────────────────────
questions = [
    "json怎么解析？",
    "怎么处理大模型输出不稳定的问题？",
    "数据清洗怎么处理空值？",
]
for q in questions:
    print(f"问题:{q}")
    answer = chain.invoke(q)
    print(f"回答:{answer}")
    print("="*50)