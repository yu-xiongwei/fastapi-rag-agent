"""
rag_immigration.py — 移民业务 RAG 知识库问答系统
业务范围：香港优才 / 专才 / 高才 | 新加坡 EP / 投资移民（GIP）
技术栈：sentence-transformers / ChromaDB / 通义千问 API（OpenAI兼容）
运行方式：python rag_immigration.py
"""

import os
import getpass
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# 1. 初始化
# ============================================================
print("正在加载 Embedding 模型...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    api_key = getpass.getpass("请输入阿里云 DashScope API Key: ")
    os.environ["DASHSCOPE_API_KEY"] = api_key

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ============================================================
# 2. 知识库文档（离线阶段）
#    实际项目中可替换为从 PDF / Word 文件读取
# ============================================================
KNOWLEDGE_BASE = [

    # ── 香港优才计划 ──────────────────────────────────────────
    {
        "id": "hk_talent_001",
        "text": "香港优秀人才入境计划（优才计划）每年配额500名，申请人须年龄在18至59岁之间，学历须达到学士学位或具备同等学历，且须具备良好品格。申请人无需在港有工作聘约，通过综合计分制或成就计分制申请。",
        "source": "香港优才计划官方指引",
        "category": "香港优才"
    },
    {
        "id": "hk_talent_002",
        "text": "优才综合计分制满分225分，分为年龄、学历、工作经验、语言能力、家庭背景五个维度。通常获邀分数在80至165分之间浮动，具体视每轮评审结果而定。申请人可通过提升学历、增加工作年限、语言证书等方式提高分数。",
        "source": "香港优才评分标准说明",
        "category": "香港优才"
    },
    {
        "id": "hk_talent_003",
        "text": "优才申请所需材料清单：有效旅行证件（护照）、近期彩色相片、学历证明文件（须经公证）、专业资格证明、无犯罪记录证明、过去十年内所有居住地的生活证明、财务证明（银行存款证明）、个人简历。",
        "source": "香港优才申请材料清单",
        "category": "香港优才"
    },
    {
        "id": "hk_talent_004",
        "text": "优才成就计分制适用于在文化艺术、体育运动、烹饪艺术三个范畴中具有杰出成就的申请人，无须累积综合积分，直接由评审委员会审核。申请人须提供能证明其杰出成就的文件，例如奖项证书、媒体报道等。",
        "source": "香港优才成就计分制说明",
        "category": "香港优才"
    },

    # ── 香港专才计划 ──────────────────────────────────────────
    {
        "id": "hk_prof_001",
        "text": "输入内地人才计划（专才计划）专为内地居民来港工作而设，申请人须持有内地居民身份证，并获得香港雇主聘用，从事香港本地所欠缺的专业工作。雇主须证明在本地招聘中未能觅得合适人选。",
        "source": "香港专才计划指引",
        "category": "香港专才"
    },
    {
        "id": "hk_prof_002",
        "text": "专才计划申请材料：申请人须提交有效的内地居民身份证、学历证明（本科或以上）、工作经验证明、雇主出具的聘用信（注明职位、薪酬、工作内容）、雇主的商业登记证副本。首次获批一般给予两年逗留期限。",
        "source": "香港专才申请材料清单",
        "category": "香港专才"
    },
    {
        "id": "hk_prof_003",
        "text": "专才申请评审重点：申请人的学历和工作经验须与拟任职位直接相关；薪酬须不低于同等职位的本地市场水平；雇主须是在港合法经营的公司。IT、金融、医疗、建筑等行业申请成功率较高。",
        "source": "香港专才评审标准",
        "category": "香港专才"
    },

    # ── 香港高才通计划 ────────────────────────────────────────
    {
        "id": "hk_hktalent_001",
        "text": "香港高端人才通行证计划（高才通）于2022年12月推出，分A、B、C三类。A类：过去一年年薪达250万港元或以上；B类：全球百强大学毕业，并在过去五年内具有三年或以上工作经验；C类：全球百强大学毕业，毕业不超过三年。",
        "source": "香港高才通官方指引",
        "category": "香港高才通"
    },
    {
        "id": "hk_hktalent_002",
        "text": "高才通申请所需材料：有效旅行证件、学历证明（须为认可院校颁发）、薪酬证明（A类：雇主证明信、粮单、报税表）、工作经验证明（B类）、近期无犯罪记录。获批后可在港居住两年，期间须寻找工作或开展业务。",
        "source": "香港高才通申请材料清单",
        "category": "香港高才通"
    },
    {
        "id": "hk_hktalent_003",
        "text": "高才通认可的全球百强大学名单每年更新，以QS世界大学排名为主要参考。内地高校中清华、北大、复旦、交大等均在认可名单内。申请人须提供官方成绩单及毕业证书，文件须为英文版或附英文翻译公证件。",
        "source": "香港高才通百强大学认定标准",
        "category": "香港高才通"
    },
    {
        "id": "hk_hktalent_004",
        "text": "高才通获批后的续签要求：首次获批两年，在港期间须受雇或从事商业活动。续签时须提交在港工作或经营业务的证明，包括雇主证明信、强积金供款记录、商业登记证等，通常可续签三年。",
        "source": "香港高才通续签要求",
        "category": "香港高才通"
    },

    # ── 新加坡就业准证（EP）──────────────────────────────────
    {
        "id": "sg_ep_001",
        "text": "新加坡就业准证（Employment Pass，EP）适用于希望在新加坡工作的外籍专业人士、经理及行政人员。自2023年9月起，申请人月薪须至少达到5000新元（金融行业须至少5500新元），且须持有认可学历或专业资质。",
        "source": "新加坡MOM就业准证指引",
        "category": "新加坡EP"
    },
    {
        "id": "sg_ep_002",
        "text": "EP申请评审采用COMPASS积分框架（2023年9月起实施），包括：薪酬与本地同类职位比较（最高20分）、学历资质（最高20分）、公司多元化程度（最高20分）、支持本地就业（最高20分）四个维度，总分须达40分或以上。",
        "source": "新加坡COMPASS评分框架",
        "category": "新加坡EP"
    },
    {
        "id": "sg_ep_003",
        "text": "EP申请所需材料：个人护照、最高学历证明、工作经验证明、雇主出具的聘用信（注明职位、薪酬）、雇主的新加坡公司注册证明（UEN）。雇主须通过MOM网站在线提交申请，处理时间约3至8周。",
        "source": "新加坡EP申请材料清单",
        "category": "新加坡EP"
    },
    {
        "id": "sg_ep_004",
        "text": "EP续签须在准证到期前6个月内申请。续签评审同样采用COMPASS框架，且会参考持证人的实际薪酬增长情况和工作表现。EP有效期通常为两年（首次）或三年（续签），最长可申请至五年。",
        "source": "新加坡EP续签要求",
        "category": "新加坡EP"
    },

    # ── 新加坡全球投资者计划（GIP）──────────────────────────
    {
        "id": "sg_gip_001",
        "text": "新加坡全球投资者计划（Global Investor Programme，GIP）为有意在新加坡投资并定居的高净值人士提供永久居留权申请渠道。投资选项分三类：选项A投资250万新元在新成立或扩展业务；选项B投资250万新元于GIP基金；选项C在单一家族办公室管理至少2亿新元资产。",
        "source": "新加坡EDB GIP官方指引",
        "category": "新加坡GIP"
    },
    {
        "id": "sg_gip_002",
        "text": "GIP申请资格要求：须为成熟商人或企业家，企业年营业额须达5000万新元或以上，且须有至少三年的业务运营记录。申请人须提交详细的商业计划书，说明在新加坡的投资计划和预期创造就业人数。",
        "source": "新加坡GIP申请资格",
        "category": "新加坡GIP"
    },
    {
        "id": "sg_gip_003",
        "text": "GIP申请材料：护照、公司财务报表（近三年经审计）、个人资产证明、无犯罪记录证明、商业计划书（须详述在新加坡的投资计划）、公司股权结构证明。获批后发放五年期入境许可，须在规定期限内完成投资。",
        "source": "新加坡GIP申请材料清单",
        "category": "新加坡GIP"
    },
    {
        "id": "sg_gip_004",
        "text": "GIP常见问题：申请人须亲自参与企业经营，不接受纯投资人。家族成员（配偶及未成年子女）可一并申请PR。完成投资后须定期向EDB提交投资进展报告，通常为每两年一次，以证明持续履行投资承诺。",
        "source": "新加坡GIP常见问题",
        "category": "新加坡GIP"
    },
]

# ============================================================
# 3. 建库（离线阶段）
# ============================================================
def build_knowledge_base() -> chromadb.Collection:
    db = chromadb.Client()
    collection = db.create_collection("immigration_kb")

    texts = [doc["text"] for doc in KNOWLEDGE_BASE]
    ids   = [doc["id"]   for doc in KNOWLEDGE_BASE]
    metas = [{"source": doc["source"], "category": doc["category"]} for doc in KNOWLEDGE_BASE]

    embeddings = embed_model.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)

    print(f"知识库建立完成，共存入 {len(KNOWLEDGE_BASE)} 条文档\n")
    return collection


# ============================================================
# 4. RAG 查询（在线阶段）
# ============================================================
def rag_query(question: str, collection: chromadb.Collection, top_k: int = 3) -> dict:
    """
    完整 RAG 链路：
    问题向量化 → ChromaDB 检索 Top-K → 拼装 Prompt → 千问生成回答
    """
    # Step 1: 检索
    q_embedding = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_docs  = results["documents"][0]
    retrieved_metas = results["metadatas"][0]

    if not retrieved_docs:
        return {"answer": "知识库中未找到相关内容，请联系顾问。", "sources": []}

    # Step 2: 拼装上下文
    context_parts = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        context_parts.append(f"[来源{i+1}：{meta['source']}]\n{doc}")
    context = "\n\n".join(context_parts)

    # Step 3: 构造 Prompt
    system_prompt = (
        "你是一位专业的移民顾问助手，专注于香港和新加坡移民业务。\n"
        "请严格根据以下【知识库内容】回答客户问题，不要编造知识库中没有的信息。\n"
        "如果知识库中没有相关内容，请明确告知客户并建议咨询专业顾问。\n"
        "回答须准确、专业、简洁，适合直接告知客户。"
    )
    user_prompt = f"【知识库内容】\n{context}\n\n【客户问题】\n{question}"

    # Step 4: 调用模型
    resp = client.chat.completions.create(
        model="qwen-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    )

    return {
        "answer":  resp.choices[0].message.content,
        "sources": [m["source"] for m in retrieved_metas],
    }


# ============================================================
# 5. 测试
# ============================================================
if __name__ == "__main__":
    collection = build_knowledge_base()

    test_questions = [
        # 香港场景
        "香港高才通A类申请需要满足什么薪酬条件？",
        "优才计划的材料清单有哪些？",
        "专才计划和高才通有什么区别？",
        # 新加坡场景
        "新加坡EP的COMPASS积分怎么计算？",
        "GIP投资移民需要投资多少钱？",
        "客户想申请新加坡PR，有什么途径？",
    ]

    for q in test_questions:
        print(f"问题：{q}")
        result = rag_query(q, collection)
        print(f"回答：{result['answer']}")
        print(f"来源：{' | '.join(result['sources'])}")
        print("=" * 60)
