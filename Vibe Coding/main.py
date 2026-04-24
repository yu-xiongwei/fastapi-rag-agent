import os
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("TODO_DB_PATH", str(BASE_DIR / "todos.db")))


app = FastAPI(title="ToDo List API")


class TodoCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    completed: bool = False


class TodoUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    completed: Optional[bool] = None


class TodoItem(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    completed: bool


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_todo(row: sqlite3.Row) -> TodoItem:
    return TodoItem(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        completed=bool(row["completed"]),
    )


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                completed INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.post("/todos", response_model=TodoItem, status_code=201)
def create_todo(payload: TodoCreate) -> TodoItem:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO todos (title, description, completed) VALUES (?, ?, ?)",
            (payload.title.strip(), payload.description, int(payload.completed)),
        )
        conn.commit()
        todo_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create todo")
    return row_to_todo(row)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/todos", response_model=list[TodoItem])
def list_todos() -> list[TodoItem]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM todos ORDER BY id DESC").fetchall()
    return [row_to_todo(row) for row in rows]


@app.get("/todos/{todo_id}", response_model=TodoItem)
def get_todo(todo_id: int) -> TodoItem:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    return row_to_todo(row)


@app.put("/todos/{todo_id}", response_model=TodoItem)
def update_todo(todo_id: int, payload: TodoUpdate) -> TodoItem:
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Todo not found")

        title = payload.title.strip() if payload.title is not None else existing["title"]
        description = payload.description if payload.description is not None else existing["description"]
        completed = int(payload.completed) if payload.completed is not None else existing["completed"]

        conn.execute(
            "UPDATE todos SET title = ?, description = ?, completed = ? WHERE id = ?",
            (title, description, completed, todo_id),
        )
        conn.commit()

        updated = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()

    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to update todo")
    return row_to_todo(updated)


@app.delete("/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int) -> None:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Todo not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
