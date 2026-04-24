"""
langchain_immigration.py — 移民业务 LangChain 全链路系统
业务范围：香港优才 / 专才 / 高才通 | 新加坡 EP / GIP 投资移民
技术栈：LangChain / sentence-transformers / ChromaDB / 通义千问 API
功能：RAG知识库问答 + 文书生成 两条链路，切换模型只改一行配置
运行方式：python langchain_immigration.py
"""

import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ============================================================
# 1. 模型配置（换模型只改这里）
# ============================================================
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    api_key = getpass.getpass("请输入阿里云 DashScope API Key: ")
    os.environ["DASHSCOPE_API_KEY"] = api_key

llm = ChatOpenAI(
    model="qwen-turbo",           # ← 换模型只改这一行
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# ============================================================
# 2. 知识库文档
# ============================================================
IMMIGRATION_DOCS = [
    # ── 香港优才 ──────────────────────────────────────────────
    Document(
        page_content=(
            "香港优秀人才入境计划（优才计划）每年配额500名，申请人年龄须在18至59岁，"
            "须持有学士学位或同等学历，无需在港有工作聘约。"
            "申请途径：综合计分制（满分225分）或成就计分制（文化/体育/烹饪艺术杰出人士）。"
            "通常获邀分数在80至165分之间浮动。"
        ),
        metadata={"source": "香港优才计划官方指引", "category": "香港优才", "scheme": "hk_talent"}
    ),
    Document(
        page_content=(
            "优才申请材料清单：有效护照、近期彩色相片、学历证明（须公证）、"
            "专业资格证明、无犯罪记录证明、过去十年居住地的生活证明、"
            "财务证明（银行存款）、个人简历。预计准备周期45至60天。"
        ),
        metadata={"source": "香港优才申请材料", "category": "香港优才", "scheme": "hk_talent"}
    ),
    Document(
        page_content=(
            "优才综合计分制评分维度：年龄（最高30分）、学历（最高70分）、"
            "工作经验（最高55分）、语言能力（最高20分）、家庭背景（最高50分）。"
            "博士学历可得70分，硕士55分，学士40分。"
            "工作经验满10年以上可得40分。持有良好英文及中文能力可得满分20分。"
        ),
        metadata={"source": "香港优才评分标准", "category": "香港优才", "scheme": "hk_talent"}
    ),

    # ── 香港专才 ──────────────────────────────────────────────
    Document(
        page_content=(
            "输入内地人才计划（专才）专为内地居民在港工作而设，"
            "申请人须获香港雇主聘用，从事香港本地所欠缺的专业工作。"
            "雇主须证明在本地招聘中未能觅得合适人选。"
            "首次获批逗留期限通常为两年，之后可续签。"
        ),
        metadata={"source": "香港专才计划指引", "category": "香港专才", "scheme": "hk_prof"}
    ),
    Document(
        page_content=(
            "专才申请须提交材料：有效内地居民身份证、本科或以上学历证明、"
            "工作经验证明、雇主聘用信（注明职位薪酬工作内容）、"
            "雇主的商业登记证副本。薪酬须不低于本地市场同等职位水平。"
            "IT、金融、医疗、建筑行业申请成功率较高。"
        ),
        metadata={"source": "香港专才申请材料", "category": "香港专才", "scheme": "hk_prof"}
    ),

    # ── 香港高才通 ────────────────────────────────────────────
    Document(
        page_content=(
            "香港高端人才通行证计划（高才通）分三类：\n"
            "A类：过去一年年薪达250万港元或以上；\n"
            "B类：全球百强大学毕业，过去五年内有三年或以上工作经验；\n"
            "C类：全球百强大学毕业，毕业不超过三年。\n"
            "获批后可在港居住两年，期间须寻找工作或开展业务。"
        ),
        metadata={"source": "香港高才通官方指引", "category": "香港高才通", "scheme": "hk_hktalent"}
    ),
    Document(
        page_content=(
            "高才通申请材料：有效旅行证件、认可院校学历证明、"
            "A类须提供薪酬证明（雇主证明信、粮单、报税表）、"
            "B类须提供工作经验证明、近期无犯罪记录证明。"
            "认可百强大学以QS世界大学排名为主要参考，"
            "内地高校中清华、北大、复旦、交大等均在认可名单内。"
        ),
        metadata={"source": "香港高才通申请材料", "category": "香港高才通", "scheme": "hk_hktalent"}
    ),
    Document(
        page_content=(
            "高才通续签要求：首次获批两年，续签时须提交在港工作或经营业务证明，"
            "包括雇主证明信、强积金（MPF）供款记录、商业登记证等。"
            "通常可续签三年，长期居港满七年可申请永久居留权（香港居民身份）。"
        ),
        metadata={"source": "香港高才通续签要求", "category": "香港高才通", "scheme": "hk_hktalent"}
    ),

    # ── 新加坡 EP ─────────────────────────────────────────────
    Document(
        page_content=(
            "新加坡就业准证（EP）适用于希望在新加坡工作的外籍专业人士。"
            "自2023年9月起，申请人月薪须至少达到5000新元（金融行业5500新元）。"
            "申请评审采用COMPASS积分框架，须达40分或以上方可通过。"
            "COMPASS四维度：薪酬比较（最高20分）、学历资质（最高20分）、"
            "公司多元化（最高20分）、支持本地就业（最高20分）。"
        ),
        metadata={"source": "新加坡EP指引", "category": "新加坡EP", "scheme": "sg_ep"}
    ),
    Document(
        page_content=(
            "EP申请所需材料：个人护照、最高学历证明、工作经验证明、"
            "雇主出具的聘用信（注明职位薪酬）、雇主新加坡公司注册证明（UEN）。"
            "雇主须通过MOM网站在线提交，处理时间约3至8周。"
            "EP有效期：首次两年，续签通常三年。"
        ),
        metadata={"source": "新加坡EP申请材料", "category": "新加坡EP", "scheme": "sg_ep"}
    ),

    # ── 新加坡 GIP ────────────────────────────────────────────
    Document(
        page_content=(
            "新加坡全球投资者计划（GIP）投资选项：\n"
            "选项A：投资250万新元在新加坡新成立或扩展业务；\n"
            "选项B：投资250万新元于GIP认可基金；\n"
            "选项C：在单一家族办公室管理至少2亿新元资产。\n"
            "申请人须为企业主或企业家，企业年营业额须达5000万新元或以上，"
            "且须有至少三年业务运营记录。"
        ),
        metadata={"source": "新加坡GIP指引", "category": "新加坡GIP", "scheme": "sg_gip"}
    ),
    Document(
        page_content=(
            "GIP申请材料：护照、近三年经审计公司财务报表、个人资产证明、"
            "无犯罪记录证明、商业计划书（须详述在新加坡投资计划及预期创造就业人数）、"
            "公司股权结构证明。获批后发放五年期入境许可，须在规定期限内完成投资。"
            "家族成员（配偶及未成年子女）可一并申请PR。"
        ),
        metadata={"source": "新加坡GIP申请材料", "category": "新加坡GIP", "scheme": "sg_gip"}
    ),
]


# ============================================================
# 3. 建向量库
# ============================================================
def build_vectorstore() -> Chroma:
    print("正在构建移民业务知识库...")
    vectorstore = Chroma.from_documents(IMMIGRATION_DOCS, embeddings)
    print(f"知识库构建完成，共存入 {len(IMMIGRATION_DOCS)} 条文档\n")
    return vectorstore


# ============================================================
# 4. 链路一：RAG 知识库问答链
# ============================================================
def build_rag_chain(vectorstore: Chroma):
    """
    RAG 问答链：用户提问 → 检索相关政策文档 → 生成专业回答
    链路：retriever | format_docs | prompt | llm | StrOutputParser
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(
            f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        )

    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一位专业的移民顾问助手，专注于香港和新加坡移民业务。\n"
            "请严格根据以下【知识库内容】回答客户问题，不要编造知识库中没有的信息。\n"
            "如果知识库中没有相关内容，请明确告知客户并建议咨询专业顾问。\n"
            "回答须准确、专业、简洁，适合直接告知客户。\n\n"
            "【知识库内容】\n{context}"
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================
# 5. 链路二：文书生成链（多场景模板）
# ============================================================
DOCUMENT_TEMPLATES = {

    "hk_application_plan": ChatPromptTemplate.from_messages([
        (
            "system",
            "你是专业的香港移民申请计划书撰写顾问。根据客户信息，生成一份专业的申请计划书。\n"
            "计划书须包含以下段落，用中文撰写，语言正式专业：\n"
            "1. 申请人基本情况介绍\n"
            "2. 申请动机及来港计划\n"
            "3. 个人专业背景及成就\n"
            "4. 对香港的贡献及价值\n"
            "5. 家庭情况及安家计划\n"
            "6. 财务状况说明"
        ),
        ("human", "请根据以下客户信息生成申请计划书：\n{client_info}")
    ]),

    "hk_supplement_statement": ChatPromptTemplate.from_messages([
        (
            "system",
            "你是专业的香港移民文书顾问。根据客户提供的补件原因，生成一份正式的补件说明信。\n"
            "说明信须包含：说明缺失文件的原因、替代证明材料说明、声明信息真实性。\n"
            "语言须正式、诚恳，符合移民局文件要求。"
        ),
        ("human", "补件情况：{client_info}\n请生成补件说明信。")
    ]),

    "sg_gip_business_plan": ChatPromptTemplate.from_messages([
        (
            "system",
            "你是专业的新加坡GIP商业计划书撰写顾问。根据客户企业背景和投资计划，"
            "生成一份符合EDB要求的商业计划书框架，用英文撰写：\n"
            "1. Executive Summary\n"
            "2. Company Background & Track Record\n"
            "3. Singapore Investment Plan\n"
            "4. Job Creation Plan\n"
            "5. Financial Projections (3-5 years)\n"
            "6. Commitment to Singapore"
        ),
        ("human", "客户企业信息：{client_info}\n请生成商业计划书。")
    ]),

    "case_social_media": ChatPromptTemplate.from_messages([
        (
            "system",
            "你是移民咨询公司的内容创作专员。根据客户成功案例信息，"
            "生成一篇适合微信公众号/小红书的自媒体文案，包含吸引人的标题、"
            "客户故事（300-500字）、案例亮点（3-5条）、话题标签（5-8个）。"
        ),
        ("human", "案例信息：{client_info}\n请生成自媒体文案。")
    ]),
}


def build_doc_chain(template_name: str):
    """
    文书生成链：客户信息 → 对应模板 → 生成文书
    """
    if template_name not in DOCUMENT_TEMPLATES:
        raise ValueError(
            f"未知文书模板：{template_name}\n"
            f"可用模板：{list(DOCUMENT_TEMPLATES.keys())}"
        )
    prompt = DOCUMENT_TEMPLATES[template_name]
    chain  = prompt | llm | StrOutputParser()
    return chain


# ============================================================
# 6. 主入口：测试两条链路
# ============================================================
if __name__ == "__main__":
    vectorstore = build_vectorstore()

    # ──────────────────────────────────────────────────────────
    # 链路一：RAG 知识库问答
    # ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("【链路一：RAG 知识库问答】")
    rag_chain = build_rag_chain(vectorstore)

    rag_questions = [
        "香港高才通B类申请需要满足什么条件？",
        "新加坡EP的COMPASS积分如何计算，满分是多少？",
        "优才综合计分制博士学历可以得多少分？",
        "GIP投资移民三个选项分别要投资多少钱？",
    ]

    for q in rag_questions:
        print(f"\n问题：{q}")
        answer = rag_chain.invoke(q)
        print(f"回答：{answer}")
        print("-" * 40)

    # ──────────────────────────────────────────────────────────
    # 链路二：文书生成
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【链路二：文书生成 - 香港申请计划书】")
    plan_chain = build_doc_chain("hk_application_plan")
    plan_result = plan_chain.invoke({
        "client_info": (
            "申请人：王先生，38岁，北京大学计算机博士，"
            "现任某互联网公司AI研究员，年薪200万人民币，"
            "已婚，配偶为医生，育有一子（6岁），"
            "申请类型：香港高才通B类，"
            "来港计划：入职香港某科技公司，子女入读国际学校。"
        )
    })
    print(plan_result)

    print("\n" + "=" * 60)
    print("【链路二：文书生成 - 新加坡案例自媒体文案】")
    social_chain = build_doc_chain("case_social_media")
    social_result = social_chain.invoke({
        "client_info": (
            "客户张女士，35岁，上海某外贸公司总经理，"
            "通过新加坡EP成功获批，全家移居新加坡，"
            "孩子在新加坡就读小学，丈夫在当地创业，"
            "客户表示非常满意新加坡的营商环境和教育体系，"
            "整个EP申请过程历时6周顺利获批。"
        )
    })
    print(social_result)

    print("\n" + "=" * 60)
    print("所有链路测试完成。换模型只需修改文件顶部 llm = ChatOpenAI(model=...) 一行。")
