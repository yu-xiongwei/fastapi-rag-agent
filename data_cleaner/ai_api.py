"""
ai_api.py — AI 数据分析后端服务
技术栈：FastAPI + pandas + OpenAI SDK（兼容千问）
运行方式：uvicorn ai_api:app --reload
"""

# ============================================================
# 1. 标准库导入
# ============================================================
import os
import json
import logging
import shutil
import sys
import getpass
from typing import Any, Dict, Optional
from rag_engine import rag_engine, ALLOWED_EXTENSIONS as RAG_EXTENSIONS

# ============================================================
# 2. 第三方库导入
# ============================================================
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

# ============================================================
# 3. 日志配置（替代 print，生产可对接日志系统）
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ai_api")

# ============================================================
# 4. 代理清除（避免本地代理干扰 API 请求）
# ============================================================
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
    os.environ.pop(_key, None)
os.environ["NO_PROXY"] = "*"

# ============================================================
# 5. 初始化 API Key 和 OpenAI 客户端（模块级，确保所有进程可用）
# ============================================================
_api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()

# 如果环境变量没有，尝试交互输入（仅在直接运行时有效）
if not _api_key and __name__ == "__main__":
    try:
        _api_key = getpass.getpass("请输入 DashScope API Key（输入不显示）: ").strip()
    except Exception:
        _api_key = input("请输入 DashScope API Key: ").strip()

if not _api_key:
    print("❌ 未提供 API Key，程序退出")
    sys.exit(1)

# 将 Key 写回环境变量，供 rag_engine 使用
os.environ["DASHSCOPE_API_KEY"] = _api_key

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
logger.info("✅ API Key 已加载，客户端初始化完成")

# ============================================================
# 6. 应用初始化 & CORS
# ============================================================
app = FastAPI(title="AI 数据分析服务", version="1.1.0")

# ⚠️ 生产环境请将 allow_origins 改为你的前端域名，例如：
# allow_origins=["http://localhost:5173", "https://your-domain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 开发阶段保持 *，上线改具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 7. 常量配置
# ============================================================
MAX_FILE_SIZE_MB = 10                          # 文件大小上限
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"} # 允许的文件类型
OUTPUT_DIR = "output"
CLEANED_DIR = "cleaned"
RAG_UPLOAD_DIR = "rag_docs"              # RAG 文档上传目录

# ============================================================
# 8. Pydantic 请求/响应模型
# ============================================================

class Question(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="用户问题")

class ReportRequest(BaseModel):
    filename: str = Field(..., min_length=1, description="已清洗文件名")

class CleanConfig(BaseModel):
    """数据清洗规则，调用方可按需传入，不传则使用默认值"""
    drop_duplicates: bool = True
    drop_all_na_rows: bool = True
    fillna_strategy: Optional[str] = Field(
        default=None,
        description="缺失值填充策略：'mean' / 'median' / 'mode' / None（直接删除含空行）"
    )
    strip_strings: bool = True

# ============================================================
# 9. 工具函数
# ============================================================

def _check_file(file: UploadFile) -> None:
    """校验文件类型和大小，不合规直接抛 HTTP 422"""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"不支持的文件类型 '{ext}'，仅接受 {sorted(ALLOWED_EXTENSIONS)}"
        )
    # 注意：UploadFile 的 size 属性在 FastAPI 0.95+ 可用；
    # 如版本较旧，可先 read() 后判断长度再 seek(0)
    if file.size and file.size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{file.size // 1024 // 1024} MB），上限 {MAX_FILE_SIZE_MB} MB"
        )


def _read_file_to_df(file: UploadFile) -> pd.DataFrame:
    """根据扩展名用对应方法读取文件为 DataFrame"""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext == ".csv":
        return pd.read_csv(file.file)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file.file)
    raise HTTPException(status_code=422, detail=f"无法解析文件类型：{ext}")


def clean_dataframe(df: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    """按配置清洗 DataFrame"""
    if config.drop_duplicates:
        df = df.drop_duplicates()

    if config.drop_all_na_rows:
        df = df.dropna(how="all")

    if config.fillna_strategy == "mean":
        df = df.fillna(df.select_dtypes(include="number").mean())
    elif config.fillna_strategy == "median":
        df = df.fillna(df.select_dtypes(include="number").median())
    elif config.fillna_strategy == "mode":
        df = df.fillna(df.mode().iloc[0])
    elif config.fillna_strategy is None:
        # 默认：不主动填充（保留 NaN，由报告分析）
        pass

    if config.strip_strings:
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str).str.strip()

    return df


def serialize_for_json(data: Any) -> Any:
    """
    递归将 pandas/numpy 类型转为 JSON 兼容的 Python 基础类型。
    关键原则：先判断"多元素类型"，再判断"单元素类型"，避免歧义。
    """
    if isinstance(data, (np.ndarray, pd.Series)):
        return [serialize_for_json(item) for item in data.tolist()]
    elif isinstance(data, pd.DataFrame):
        return serialize_for_json(data.to_dict())
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    elif isinstance(data, dict):
        return {k: serialize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_for_json(i) for i in data]
    # 对单个标量安全判断 NaN（仅 int/float/str 可能为 NaN）
    elif isinstance(data, (int, float)):
        return None if pd.isna(data) else data
    return data


import json
import re
from typing import Dict, Any, Optional

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """防御性 JSON 解析，处理 LLM 输出的各种格式异常"""
    if not text:
        return None

    # 第一层：最理想情况，直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 第二层：正则抠出 {...}（处理前后有废话的情况）
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    # 第三层：去掉 Markdown 代码块标记再抠
    try:
        # 移除 ```json 和 ``` 标记
        text_clean = re.sub(r'```json\s*|```', '', text)
        match = re.search(r'\{.*\}', text_clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    # 第四层：修复中文标点再抠（处理中文冒号、中文引号等情况）
    try:
        # 替换中文标点为英文标点
        text_fix = text.replace('：', ':').replace('“', '"').replace('”', '"')
        match = re.search(r'\{.*\}', text_fix, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    # 全部失败，返回 None（绝不抛异常导致程序崩溃）
    return None

# ============================================================
# 10. 路由接口
# ============================================================

@app.get("/", summary="健康检查")
def health_check():
    return {"status": "ok", "message": "AI 数据分析服务运行正常"}


@app.post("/ask", summary="AI 对话")
def ask(question: Question):
    """发送一条问题，返回 AI 的简短回答"""
    try:
        resp = client.chat.completions.create(
            model="qwen-turbo",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个简洁、专业的 AI 助手。回答限 1~2 句，不解释，直接给答案。"
                },
                {"role": "user", "content": question.text},
            ],
        )
        return {"answer": resp.choices[0].message.content}
    except Exception as e:
        logger.error("AI 对话失败：%s", e)
        raise HTTPException(status_code=502, detail=f"AI 调用失败：{e}")


@app.post("/data/clean", summary="上传并清洗数据文件（CSV / Excel）")
async def clean_data(
    file: UploadFile = File(...),
    drop_duplicates: bool = True,
    drop_all_na_rows: bool = True,
    fillna_strategy: Optional[str] = None,
    strip_strings: bool = True,
):
    """
    上传 CSV 或 Excel 文件，执行数据清洗后返回统计信息并保存清洗结果。
    Query 参数即清洗规则，例如 ?fillna_strategy=mean
    """
    _check_file(file)
    config = CleanConfig(
        drop_duplicates=drop_duplicates,
        drop_all_na_rows=drop_all_na_rows,
        fillna_strategy=fillna_strategy,
        strip_strings=strip_strings,
    )
    try:
        df = _read_file_to_df(file)
        total_before = len(df)
        df_cleaned = clean_dataframe(df, config)
        total_after = len(df_cleaned)

        os.makedirs(CLEANED_DIR, exist_ok=True)
        out_path = os.path.join(CLEANED_DIR, file.filename)
        # 统一以 CSV 保存，方便后续报告读取
        df_cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")

        logger.info("清洗完成：%s，%d → %d 行", file.filename, total_before, total_after)
        return {
            "success": True,
            "filename": file.filename,
            "total_before": total_before,
            "total_after": total_after,
            "removed": total_before - total_after,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("数据清洗失败：%s", e)
        raise HTTPException(status_code=500, detail=f"数据清洗失败：{e}")


@app.post("/generate/report", summary="对已清洗文件生成 AI 质量报告")
def generate_report(req: ReportRequest):
    """读取已清洗文件，调用 AI 生成 JSON 格式的数据质量报告"""
    csv_path = os.path.join(CLEANED_DIR, req.filename)
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"文件不存在：{csv_path}，请先调用 /data/clean")

    try:
        df = pd.read_csv(csv_path)
        summary = {
            "总行数": int(len(df)),
            "总列数": int(len(df.columns)),
            "列名": list(df.columns),
            "缺失值统计": {col: int(df[col].isnull().sum()) for col in df.columns},
            "数据类型": {col: str(df[col].dtype) for col in df.columns},
            "重复行数量": int(df.duplicated().sum()),
        }
        summary = serialize_for_json(summary)

        resp = client.chat.completions.create(
            model="qwen-turbo",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是数据质量分析师。只输出合法 JSON，不输出任何其他文字。"
                        '格式：{"quality_score":<0-100整数>,"issues":[<问题列表>],'
                        '"recommendations":[<建议列表>],"is_production_ready":<true|false>}'
                    ),
                },
                {"role": "user", "content": f"请分析以下数据摘要：\n{json.dumps(summary, ensure_ascii=False, indent=2)}"},
            ],
        )
        result = safe_json_parse(resp.choices[0].message.content)

        for field in ("quality_score", "issues", "recommendations", "is_production_ready"):
            if field not in result:
                raise ValueError(f"模型输出缺少字段：{field}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = os.path.join(OUTPUT_DIR, f"{req.filename}_report.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(serialize_for_json(result), f, ensure_ascii=False, indent=2)

        logger.info("报告生成完成：%s", out_file)
        return {
            "success": True,
            "report": {
                "quality_score": int(result["quality_score"]),
                "issues": list(result["issues"]),
                "recommendations": list(result["recommendations"]),
                "is_production_ready": bool(result["is_production_ready"]),
            },
            "summary": summary,
            "output_path": out_file,
        }
    except HTTPException:
        raise
    except ValueError as e:
        logger.error("报告解析失败：%s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("报告生成失败：%s", e)
        raise HTTPException(status_code=502, detail=f"报告生成失败：{e}")


# ============================================================
# RAG 接口 1：上传文档并向量化
# ============================================================
@app.post("/rag/upload", summary="上传文档并向量化存入知识库")
async def rag_upload(file: UploadFile = File(...)):
    """
    支持格式：TXT / PDF / Markdown
    流程：保存文件 → 解析分块 → 向量化 → 存入 ChromaDB
    """
    import shutil
    from rag_engine import rag_engine, ALLOWED_EXTENSIONS as RAG_EXTENSIONS

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in RAG_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"不支持的格式 '{ext}'，仅接受 {sorted(RAG_EXTENSIONS)}"
        )

    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(RAG_UPLOAD_DIR, file.filename)

    try:
        # 保存上传文件到本地
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 向量化存储
        result = rag_engine.add_document(save_path, file.filename)
        logger.info("RAG 文档上传成功：%s", file.filename)
        return {"success": True, **result}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("RAG 上传失败：%s", e)
        raise HTTPException(status_code=500, detail=f"向量化失败：{e}")


# ============================================================
# RAG 接口 2：基于知识库的问答
# ============================================================
@app.post("/rag/ask", summary="基于知识库检索 + AI 回答")
def rag_ask(question: Question):
    """
    流程：问题向量化 → ChromaDB 检索相关片段 → 拼接 Prompt → 千问生成回答
    """
    from rag_engine import rag_engine

    try:
        # Step 1：检索相关片段
        retrieved = rag_engine.retrieve(question.text, top_k=3)

        if not retrieved:
            return {
                "answer": "知识库暂无相关文档，请先通过 /rag/upload 上传文档。",
                "sources": [],
            }

        # Step 2：拼接检索结果作为上下文
        context = "\n\n---\n\n".join(
            f"【来源：{r['filename']}（相关度 {r['score']}）】\n{r['text']}"
            for r in retrieved
        )

        # Step 3：构造 RAG Prompt
        system_prompt = (
            "你是一个专业的问答助手。请严格基于以下【知识库内容】回答用户问题。\n"
            '如果知识库中没有相关信息，请明确说明"知识库中未找到相关内容"，不要编造。\n'
            "回答要简洁、准确，可适当引用来源文件名。"
        )
        user_prompt = f"【知识库内容】\n{context}\n\n【用户问题】\n{question.text}"

        # Step 4：调用千问生成回答
        resp = client.chat.completions.create(
            model="qwen-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = resp.choices[0].message.content

        return {
            "answer": answer,
            "sources": [
                {"filename": r["filename"], "score": r["score"], "snippet": r["text"][:100] + "..."}
                for r in retrieved
            ],
        }

    except Exception as e:
        logger.error("RAG 问答失败：%s", e)
        raise HTTPException(status_code=502, detail=f"RAG 问答失败：{e}")


# ============================================================
# RAG 接口 3：查看知识库中已上传的文档列表
# ============================================================
@app.get("/rag/docs", summary="查看知识库中已上传的文档列表")
def rag_list_docs():
    """返回当前 ChromaDB 中已向量化的所有文档名"""
    from rag_engine import rag_engine

    try:
        docs = rag_engine.list_documents()
        return {"total": len(docs), "documents": docs}
    except Exception as e:
        logger.error("获取文档列表失败：%s", e)
        raise HTTPException(status_code=500, detail=f"获取列表失败：{e}")


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 FastAPI 服务，监听 http://127.0.0.1:8000")
    uvicorn.run("ai_api:app", host="127.0.0.1", port=8000, reload=False)