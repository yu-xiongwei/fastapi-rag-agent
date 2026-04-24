from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_qwen_api_key():
    api_key = os.getenv("QWEN_API_KEY")
    if api_key:
        return api_key
    
    while True:
        print("="*50)
        print("⚠️  未检测到千问API Key环境变量！")
        print("请手动输入你的千问API Key（输入后按回车确认）：")
        print("="*50)
        
        try:
            import getpass
            user_input = getpass.getpass("👉 千问API Key：").strip()
        except ImportError:
            user_input = input("👉 千问API Key：").strip()
        
        if user_input:
            print(f"\n✅ 已收到API Key（前8位：{user_input[:8]}...），正在验证有效性...")
            if user_input.startswith("sk-"):
                return user_input
            else:
                print(" 输入的API Key格式错误！千问API Key通常以 sk- 开头，请检查后重新输入。\n")
        else:
            print(" 未输入任何内容，请重新输入！\n")

QWEN_API_KEY = get_qwen_api_key()
print(f"\n API Key验证通过，正在启动FastAPI服务...")

QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" 

@app.post("/generate-copy")
async def generate_copy(data: dict):
    required = ["name", "features", "keywords"]
    for key in required:
        if not data.get(key):
            raise HTTPException(status_code=400, detail=f"缺少必填参数：{key}")
    
    tone = data.get("tone", "professional and trustworthy")
    audience = data.get("audience", "")
    comp = data.get("comp", "")
    
    system_prompt = f"""你是跨境电商领域的亚马逊Listing文案专家。
仅输出有效的JSON格式内容（不包含Markdown代码围栏、无多余文本），需符合以下结构：
{{
  "title": "产品标题字符串，最多200个字符，核心关键词靠近开头",
  "bullets": ["卖点1","卖点2","卖点3","卖点4","卖点5"],
  "description": "产品简短描述，150-300个字符，纯散文风格（无格式）"
}}
语气要求：{tone}。所有文案内容需同时生成中文和英文版本（例如标题包含中英文、卖点和描述均为中英文对照）。"""

    user_prompt = f"""为以下产品生成Listing文案：
产品名称：{data['name']}
产品功能：{data['features']}
核心关键词：{data['keywords']}
{audience and f"目标受众：{audience}" or ""}
{comp and f"竞品备注：{comp}" or ""}
仅返回JSON格式内容，确保文案包含中英文双语版本。"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                QWEN_API_URL,
                headers={
                    "Authorization": f"Bearer {QWEN_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen-turbo",
                    "messages": [ 
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 1000,  
                    "temperature": 0.7
                }
            )
        response.raise_for_status()
        qwen_res = response.json()
        
        raw_text = qwen_res["choices"][0]["message"]["content"].strip().replace("```json", "").replace("```", "")  
        parsed_data = json.loads(raw_text)
        return parsed_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)