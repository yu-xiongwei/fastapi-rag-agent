from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/query")
def query(text: str):
    return {"input": text, "result": f"处理了: {text}"}