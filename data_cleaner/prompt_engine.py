from openai import OpenAI

class PromptEngine:
    """可复用的 Prompt 模板管理器"""
    
    # 所有模板集中在这里管理
    TEMPLATES = {
        "data_analysis": {
            "system": "你是数据质量分析师。只输出合法 JSON，不输出任何其他文字。格式：{\"quality_score\": <0-100整数>, \"issues\": [<问题列表>], \"recommendations\": [<建议列表>], \"is_production_ready\": <true/false>}。评判标准：缺失率>5%列入issues，重复率>1%列入issues，每个issue扣10分。",
            "temperature": 0
        },
        "code_review": {
            "system": "你是资深Python工程师。只输出合法 JSON。格式：{\"score\": <0-100整数>, \"bugs\": [<bug描述列表>], \"suggestions\": [<优化建议列表>]}",
            "temperature": 0
        },
        "error_diagnosis": {
            "system": "你是Python报错专家。只输出合法 JSON。格式：{\"error_type\": <错误类型字符串>, \"root_cause\": <根本原因字符串>, \"fix\": <修复方案字符串>}",
            "temperature": 0
        }
    }
    
    def __init__(self, client: OpenAI, model: str = "qwen-turbo"):
        self.client = client
        self.model = model
    
    def run(self, template_name: str, user_message: str) -> str:
        """按模板名调用，返回模型原始输出"""
        if template_name not in self.TEMPLATES:
            raise ValueError(f"未知模板：{template_name}，可用模板：{list(self.TEMPLATES.keys())}")
        
        template = self.TEMPLATES[template_name]
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=template["temperature"],
            messages=[
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content


# ── 测试三种模板 ──────────────────────────────
if __name__ == "__main__":
    import json
    
    client = OpenAI(
        api_key="sk-4becd4ec98e6435293b76cb8ed7fbcaf",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    engine = PromptEngine(client)
    
    # 测试1：数据分析
    result1 = engine.run("data_analysis", "数据集共500行，缺失率8%，重复率2%，主键无空值")
    print("数据分析结果：")
    print(json.loads(result1))
    print()
    
    # 测试2：代码审查
    result2 = engine.run("code_review", "请审查：result = json.loads(report)")
    print("代码审查结果：")
    print(json.loads(result2))
    print()
    
    # 测试3：错误诊断
    result3 = engine.run("error_diagnosis", "报错：NameError: name 'repot' is not defined")
    print("错误诊断结果：")
    print(json.loads(result3))