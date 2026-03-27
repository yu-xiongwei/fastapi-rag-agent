import json
from openai import OpenAI
from safe_json_parse import safe_json_parse

# ── 1. 清洗报告数据（从 telco_cleaner.py 的结果来）──────
clean_summary = {
    "数据集": "Telco Customer Churn",
    "原始行数": 7043,
    "清洗后行数": 7043,
    "发现问题": [
        "TotalCharges 字段类型错误： 字符串存数字",
        "TotalCharges 含11行空格伪装的空值,isnull()无法直接监测",
        "17列字符串字段存在首尾空格",

    ],
    "数据可用率": "100%",
    "字段完整率": "100%"

}

summary_text = json.dumps(clean_summary, ensure_ascii=False, indent=2)

# ── 2. 调用大模型 API ──────────────────────────────────
client = OpenAI(
    api_key="sk-4becd4ec98e6435293b76cb8ed7fbcaf",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-turbo",
    temperature=0,
    messages=[
        {
            "role":"system",
            "content":"""你是数据质量分析师。只输出合法JSON,不输出任何其他文字。格式如下：
            {
            "quality_score":<0-100整数>,
            "risk_level":<"低风险"或"中风险"或"高风险">,
            "issues_summary": [<问题描述字符串列表>],
            "recommendations": [<改进建议字符串列表>],
            "is_production_ready":<true或false>
            }
            评判标准： 有隐蔽空值扣15分,有类型错误扣10分,有字符串污染扣5分,从100分开始扣。"""

        },
        {
            "role":"user",
            "content":f"请分析以下数据清洗报告:\n{summary_text}"
        }
    ]
)


# ── 3. 解析并打印结果 ──────────────────────────────────
report = response.choices[0].message.content
print("模型原始输出:")
print(report)
print()

result = safe_json_parse(report)
print("="*50)
print("📊 AI 数据质量分析报告")
print("=" * 50)
print(f"质量分数:  {result['quality_score']}")
print(f"风险等级：   {result['risk_level']}")
print(f"问题摘要：   {result['issues_summary']}")
print(f"改进建议：   {result['recommendations']}")
print(f"可上线：     {result['is_production_ready']}")