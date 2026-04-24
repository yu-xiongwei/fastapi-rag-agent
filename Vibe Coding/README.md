# ToDo List API (FastAPI + SQLite)

## 1. 本地环境准备（Windows）

```powershell
cd "D:\PyDemo\Vibe Coding"

# 创建虚拟环境（推荐 Python 3.11）
py -3.11 -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1
```

如果激活时报错 `running scripts is disabled`，先执行一次：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 2. 安装依赖

```powershell
pip install -r requirements.txt
```

## 3. 启动服务

方式 A（推荐）：

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

方式 B（直接运行 Python 文件）：

```powershell
python main.py
```

## 4. 接口说明

- `POST /todos`：新增待办
- `GET /todos`：查询全部待办
- `GET /todos/{todo_id}`：查询单条待办
- `PUT /todos/{todo_id}`：修改待办
- `DELETE /todos/{todo_id}`：删除待办
- `GET /health`：健康检查

启动后可访问 Swagger 文档：

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

SQLite 数据库文件会自动生成在当前目录：

- `todos.db`

你也可以通过环境变量指定数据库路径（可选）：

```powershell
$env:TODO_DB_PATH = "D:\data\todo.db"
```

如果要让同一局域网设备访问服务，可改为：

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000


```

 <!-- 建立了基于 Cursor + Python 的 AI 协作开发环境，已跑通基本对话生成能力 -->
