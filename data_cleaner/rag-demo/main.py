import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ① 配置API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-a0bfebd796a24e2f90c0f71a94777718"
# 清除代理设置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# ② 加载文档
print("正在加载文档...")
loader = TextLoader("docs/notes.txt", encoding="utf-8")
documents = loader.load()
print(f"加载完成，共 {len(documents)} 个文档")

# ③ 切割chunk
print("正在切割chunk...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)
print(f"切割完成，共 {len(chunks)} 个chunk")

# ④ 向量化 + 存入本地向量库
print("正在向量化并存入Chroma...")
embedding = DashScopeEmbeddings(model="text-embedding-v1")
vectorstore = Chroma.from_documents(chunks, embedding, persist_directory="./chroma_db")
print("向量库构建完成")

# ⑤ 构建RAG问答链（新写法）
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatTongyi(model_name="qwen-turbo")

prompt = ChatPromptTemplate.from_template("""
你是一个问答助手，请根据以下上下文回答问题。
如果上下文中没有相关信息，请说"我在文档中找不到相关信息"。

上下文：
{context}

问题：{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ⑥ 开始问答
print("\n=== RAG问答系统启动 ===\n")
while True:
    question = input("请输入你的问题（输入q退出）：")
    if question == "q":
        break
    answer = chain.invoke(question)
    print(f"\n答案：{answer}\n")