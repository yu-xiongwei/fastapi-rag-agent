from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ── 1. 初始化模型 ──────────────────────────────────
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key="sk-4becd4ec98e6435293b76cb8ed7fbcaf",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

# ── 2. 定义两个工具 ────────────────────────────────
@tool
def data_analysis_tool(question: str) -> str:
    """处理数据分析、数据清洗、pandas相关问题"""
    return f"[数据分析工具] 针对「{question}」：建议使用pandas处理，注意检查空值和类型错误。"

@tool
def code_review_tool(question: str) -> str:
    """处理代码审查、bug排查、Python语法相关问题"""
    return f"[代码审查工具] 针对「{question}」：建议检查异常处理和边界条件。"

# ── 3. 把工具绑定给模型 ────────────────────────────
tools = [data_analysis_tool, code_review_tool]
llm_with_tools = llm.bind_tools(tools)

# ── 4. 测试 ────────────────────────────────────────
questions = [
    "我的DataFrame有空值怎么处理？",
    "我的Python代码报了IndexError怎么排查？",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"问题：{q}")
    response = llm_with_tools.invoke([HumanMessage(content=q)])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Agent 选择工具：{tool_call['name']}")
        
        # 执行工具
        if tool_call['name'] == 'data_analysis_tool':
            result = data_analysis_tool.invoke(tool_call['args'])
        else:
            result = code_review_tool.invoke(tool_call['args'])
        print(f"工具返回：{result}")
    else:
        print(f"直接回答：{response.content}")