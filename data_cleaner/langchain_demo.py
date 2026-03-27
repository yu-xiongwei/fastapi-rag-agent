from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 1. 初始化大模型 ────────────────────────────────
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key="sk-4becd4ec98e6435293b76cb8ed7fbcaf",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

# ── 2. 定义 Prompt 模板 ───────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是数据质量分析师，回答简洁专业。"),
    ("user", "{question}")
])

# ── 3. 用 | 把组件串联成 Chain ─────────────────────
chain = prompt | llm | StrOutputParser()

# ── 4. 测试三个问题 ────────────────────────────────
questions = [
    "json.loads()和json.dumps()有什么区别？",
    "temperature=0有什么作用？",
    "什么是RAG？",
]

for q in questions:
    print(f"问题：{q}")
    answer = chain.invoke({"question": q})
    print(f"回答：{answer}")
    print("-" * 50)