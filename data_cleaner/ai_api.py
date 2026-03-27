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
from typing import Any, Dict, Optional

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
# 5. 应用初始化 & CORS
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
# 6. OpenAI 客户端（API Key 从环境变量读取，不硬编码）
# ============================================================
# 启动前在终端执行：set DASHSCOPE_API_KEY=sk-xxxxxx  (Windows)
#                   export DASHSCOPE_API_KEY=sk-xxxxxx (Mac/Linux)
_api_key = os.environ.get("DASHSCOPE_API_KEY")
if not _api_key:
    # 开发阶段给出明确提示，而不是静默失败
    logger.warning("⚠️  未设置环境变量 DASHSCOPE_API_KEY，AI 功能将不可用！")

client = OpenAI(
    api_key=_api_key or "MISSING_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
# 7. 常量配置
# ============================================================
MAX_FILE_SIZE_MB = 10                          # 文件大小上限
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"} # 允许的文件类型
OUTPUT_DIR = "output"
CLEANED_DIR = "cleaned"

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


def safe_json_parse(text: str) -> Dict[str, Any]:
    """剥离模型可能输出的 Markdown 代码块，再解析 JSON"""
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if l.strip()]
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].endswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败：{e.msg}，原文片段：{text[:200]}") from e

# ============================================================
# 10. 路由接口
# ============================================================

@app.get("/", summary="健康检查")
def health_check():
    return {"status": "ok", "message": "AI 数据分析服务运行正常"}


@app.post("/ask", summary="AI 对话")
def ask(question: Question):
    """发送一条问题，返回 AI 的简短回答"""
    if not _api_key:
        raise HTTPException(status_code=503, detail="AI 服务未配置（缺少 DASHSCOPE_API_KEY）")
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
    if not _api_key:
        raise HTTPException(status_code=503, detail="AI 服务未配置（缺少 DASHSCOPE_API_KEY）")

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
