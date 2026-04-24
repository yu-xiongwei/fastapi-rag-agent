"""
prompt_engine_immigration.py — 移民业务多场景 Prompt 模板管理器
业务范围：香港优才 / 专才 / 高才通 | 新加坡 EP / GIP 投资移民
技术栈：通义千问 API（OpenAI兼容）/ Python
运行方式：python prompt_engine_immigration.py
"""

import os
import json
import re
import getpass
import logging
from typing import Optional, Dict, Any
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prompt_engine_immigration")

# ============================================================
# 安全 JSON 解析（防御模型输出格式异常）
# ============================================================
def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    try:
        clean = re.sub(r'```json\s*|```', '', text)
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None


# ============================================================
# PromptEngine 核心类
# ============================================================
class ImmigrationPromptEngine:
    """
    移民业务多场景 Prompt 模板管理器
    支持场景：材料清单生成、计划书生成、案例文案生成、补件声明生成、资质评估
    """

    TEMPLATES = {

        # ── 香港优才：材料清单生成 ─────────────────────────────
        "hk_talent_checklist": {
            "description": "香港优才计划申请材料清单生成",
            "system": (
                "你是香港优才计划专业顾问。根据客户信息，生成一份完整的申请材料清单。\n"
                "只输出合法 JSON，不输出任何其他文字。\n"
                "格式：{\n"
                '  "scheme": "香港优才计划",\n'
                '  "applicant_type": <综合计分制或成就计分制>,\n'
                '  "required_documents": [\n'
                '    {"document": <材料名称>, "requirement": <具体要求>, "notes": <注意事项>}\n'
                "  ],\n"
                '  "estimated_preparation_days": <预计准备天数>,\n'
                '  "special_reminders": [<特别提示列表>]\n'
                "}"
            ),
            "temperature": 0,
            "output_format": "json"
        },

        # ── 香港专才：材料清单生成 ─────────────────────────────
        "hk_prof_checklist": {
            "description": "香港专才计划申请材料清单生成",
            "system": (
                "你是香港输入内地人才计划（专才）专业顾问。根据客户信息，生成完整申请材料清单。\n"
                "只输出合法 JSON，不输出任何其他文字。\n"
                "格式：{\n"
                '  "scheme": "香港专才计划",\n'
                '  "required_documents": [\n'
                '    {"document": <材料名称>, "requirement": <具体要求>, "notes": <注意事项>}\n'
                "  ],\n"
                '  "employer_documents": [<雇主须提交的材料列表>],\n'
                '  "estimated_preparation_days": <预计准备天数>,\n'
                '  "special_reminders": [<特别提示>]\n'
                "}"
            ),
            "temperature": 0,
            "output_format": "json"
        },

        # ── 香港高才通：资质评估 ───────────────────────────────
        "hk_hktalent_assessment": {
            "description": "香港高才通申请资质评估",
            "system": (
                "你是香港高端人才通行证计划（高才通）专业顾问。根据客户信息，评估其申请资质并推荐适合的类别。\n"
                "只输出合法 JSON，不输出任何其他文字。\n"
                "格式：{\n"
                '  "scheme": "香港高才通",\n'
                '  "recommended_category": <A类/B类/C类>,\n'
                '  "eligibility_score": <资质评分 0-100>,\n'
                '  "strengths": [<优势列表>],\n'
                '  "weaknesses": [<劣势/风险列表>],\n'
                '  "success_probability": <成功概率百分比>,\n'
                '  "recommended_action": <建议行动方案>\n'
                "}"
            ),
            "temperature": 0,
            "output_format": "json"
        },

        # ── 香港：申请计划书生成（文书）─────────────────────────
        "hk_application_plan": {
            "description": "香港移民申请计划书（文书）生成",
            "system": (
                "你是专业的香港移民申请计划书撰写顾问。根据客户信息，生成一份专业的申请计划书文本。\n"
                "计划书须包含以下段落，用中文撰写，语言正式专业：\n"
                "1. 申请人基本情况介绍\n"
                "2. 申请动机及来港计划\n"
                "3. 个人专业背景及成就\n"
                "4. 对香港的贡献及价值\n"
                "5. 家庭情况及安家计划\n"
                "6. 财务状况说明\n"
                "直接输出计划书正文，不需要 JSON 格式，语言须正式、真实、有说服力。"
            ),
            "temperature": 0.3,
            "output_format": "text"
        },

        # ── 香港：补件声明生成 ─────────────────────────────────
        "hk_supplement_statement": {
            "description": "香港移民申请补件声明/解释信生成",
            "system": (
                "你是专业的香港移民文书顾问。根据客户提供的补件原因，生成一份正式的补件说明信。\n"
                "说明信须包含：说明缺失文件的原因、替代证明材料说明、声明信息真实性。\n"
                "语言须正式、诚恳，符合移民局文件要求。直接输出信件正文。"
            ),
            "temperature": 0.2,
            "output_format": "text"
        },

        # ── 香港：客户成功案例自媒体文案 ─────────────────────────
        "hk_case_social_media": {
            "description": "香港移民成功案例自媒体文案生成",
            "system": (
                "你是移民咨询公司的内容创作专员。根据客户成功案例信息，生成一篇适合微信公众号/小红书的自媒体文案。\n"
                "只输出合法 JSON，不输出任何其他文字。\n"
                "格式：{\n"
                '  "title": <吸引眼球的标题>,\n'
                '  "hook": <开篇吸引读者的1-2句话>,\n'
                '  "story": <客户故事正文，300-500字，真实感强>,\n'
                '  "key_points": [<3-5个案例亮点，适合做成要点列表>],\n'
                '  "cta": <结尾的引导语/行动召唤>,\n'
                '  "hashtags": [<5-8个适合的话题标签>]\n'
                "}"
            ),
            "temperature": 0.7,
            "output_format": "json"
        },

        # ── 新加坡 EP：材料清单生成 ────────────────────────────
        "sg_ep_checklist": {
            "description": "新加坡就业准证（EP）申请材料清单生成",
            "system": (
                "你是新加坡就业准证（Employment Pass）专业顾问。根据客户信息，生成完整申请材料清单。\n"
                "只输出合法 JSON，不输出任何其他文字。\n"
                "格式：{\n"
                '  "scheme": "新加坡就业准证（EP）",\n'
                '  "compass_pre_assessment": {\n'
                '    "estimated_score": <COMPASS预估分数>,\n'
                '    "pass_threshold": 40,\n'
                '    "assessment": <通过/风险/不通过>\n'
                '  },\n'
                '  "required_documents": [\n'
                '    {"document": <材料名称>, "requirement": <具体要求>, "notes": <注意事项>}\n'
                "  ],\n"
                '  "employer_responsibilities": [<雇主须完成的事项>],\n'
                '  "estimated_processing_weeks": <预计处理周数>,\n'
                '  "special_reminders": [<特别提示>]\n'
                "}"
            ),
            "temperature": 0,
            "output_format": "json"
        },

        # ── 新加坡 GIP：资质评估 ───────────────────────────────
        "sg_gip_assessment": {
            "description": "新加坡全球投资者计划（GIP）申请资质评估",
            "system": (
                "你是新加坡全球投资者计划（GIP）专业顾问。根据客户背景，评估其申请资质并推荐最合适的投资选项。\n"
                "只输出合法 JSON，不输出任何其他文字。\n"
                "格式：{\n"
                '  "scheme": "新加坡GIP",\n'
                '  "recommended_option": <选项A/B/C>,\n'
                '  "option_details": <推荐选项的具体投资要求说明>,\n'
                '  "eligibility_assessment": {\n'
                '    "meets_revenue_threshold": <true/false>,\n'
                '    "meets_operating_history": <true/false>,\n'
                '    "asset_sufficiency": <true/false>\n'
                '  },\n'
                '  "strengths": [<申请优势>],\n'
                '  "risks": [<申请风险/不足>],\n'
                '  "success_probability": <成功概率百分比>,\n'
                '  "next_steps": [<建议的后续步骤>]\n'
                "}"
            ),
            "temperature": 0,
            "output_format": "json"
        },

        # ── 新加坡 GIP：商业计划书生成 ────────────────────────
        "sg_gip_business_plan": {
            "description": "新加坡GIP申请商业计划书生成",
            "system": (
                "你是专业的新加坡GIP商业计划书撰写顾问。根据客户的企业背景和投资计划，生成一份符合EDB要求的商业计划书框架。\n"
                "计划书须包含以下部分，用英文撰写（符合EDB提交要求）：\n"
                "1. Executive Summary\n"
                "2. Company Background & Track Record\n"
                "3. Singapore Investment Plan\n"
                "4. Job Creation Plan (number & type of local jobs)\n"
                "5. Financial Projections (3-5 years)\n"
                "6. Commitment to Singapore\n"
                "直接输出计划书正文，语言须专业、具体、有说服力。"
            ),
            "temperature": 0.3,
            "output_format": "text"
        },

        # ── 新加坡：客户成功案例自媒体文案 ───────────────────────
        "sg_case_social_media": {
            "description": "新加坡移民成功案例自媒体文案生成",
            "system": (
                "你是移民咨询公司的内容创作专员，专注于新加坡移民案例宣传。\n"
                "根据客户成功案例信息，生成一篇适合微信公众号/小红书的自媒体文案。\n"
                "只输出合法 JSON，不输出任何其他文字。\n"
                "格式：{\n"
                '  "title": <吸引眼球的标题>,\n'
                '  "hook": <开篇吸引读者的1-2句话>,\n'
                '  "story": <客户故事正文，300-500字，突出新加坡生活优势>,\n'
                '  "key_points": [<3-5个案例亮点>],\n'
                '  "cta": <结尾引导语>,\n'
                '  "hashtags": [<5-8个话题标签>]\n'
                "}"
            ),
            "temperature": 0.7,
            "output_format": "json"
        },
    }

    def __init__(self, client: OpenAI, model: str = "qwen-turbo"):
        self.client = client
        self.model  = model

    def list_templates(self) -> None:
        print("\n可用模板列表：")
        for name, cfg in self.TEMPLATES.items():
            print(f"  [{name}] {cfg['description']}")
        print()

    def run(self, template_name: str, client_info: str) -> Dict[str, Any]:
        """
        按模板名生成内容
        :param template_name: 模板名称（见 TEMPLATES）
        :param client_info:   客户信息描述（自然语言输入）
        :return: {"raw": 原始输出, "parsed": 解析后内容（JSON模板）或 None（文本模板）}
        """
        if template_name not in self.TEMPLATES:
            raise ValueError(
                f"未知模板：{template_name}\n可用模板：{list(self.TEMPLATES.keys())}"
            )

        tmpl = self.TEMPLATES[template_name]
        logger.info("使用模板 [%s] 生成内容...", template_name)

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=tmpl["temperature"],
            messages=[
                {"role": "system", "content": tmpl["system"]},
                {"role": "user",   "content": client_info},
            ]
        )
        raw = resp.choices[0].message.content

        parsed = None
        if tmpl["output_format"] == "json":
            parsed = safe_json_parse(raw)
            if parsed is None:
                logger.warning("JSON 解析失败，返回原始文本")

        return {"raw": raw, "parsed": parsed, "template": template_name}


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("请输入阿里云 DashScope API Key: ")
        os.environ["DASHSCOPE_API_KEY"] = api_key

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    engine = ImmigrationPromptEngine(client)
    engine.list_templates()

    # ── 测试1：香港高才通资质评估 ──────────────────────────────
    print("=" * 60)
    print("测试1：香港高才通资质评估")
    result1 = engine.run(
        "hk_hktalent_assessment",
        "客户基本信息：女性，32岁，清华大学本科+硕士，计算机专业，毕业5年，"
        "目前在北京某科技公司担任高级工程师，年薪80万人民币，无犯罪记录，未婚无子女。"
    )
    if result1["parsed"]:
        print(json.dumps(result1["parsed"], ensure_ascii=False, indent=2))
    else:
        print(result1["raw"])

    # ── 测试2：新加坡EP材料清单 ───────────────────────────────
    print("=" * 60)
    print("测试2：新加坡EP材料清单")
    result2 = engine.run(
        "sg_ep_checklist",
        "客户信息：男性，35岁，复旦大学金融学本科，在新加坡某金融科技公司获得聘用，"
        "职位：高级产品经理，月薪8000新元，目前持中国护照。"
    )
    if result2["parsed"]:
        print(json.dumps(result2["parsed"], ensure_ascii=False, indent=2))
    else:
        print(result2["raw"])

    # ── 测试3：香港成功案例自媒体文案 ────────────────────────
    print("=" * 60)
    print("测试3：香港成功案例自媒体文案")
    result3 = engine.run(
        "hk_case_social_media",
        "案例信息：客户李先生，40岁，上海某外资企业财务总监，通过香港高才通A类获批，"
        "全家三口（含一个8岁儿子）成功移居香港，孩子已入读国际学校，"
        "客户表示对香港的国际化环境和教育资源非常满意，整个申请过程历时3个月。"
    )
    if result3["parsed"]:
        print(json.dumps(result3["parsed"], ensure_ascii=False, indent=2))
    else:
        print(result3["raw"])
