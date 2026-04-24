"""
rag_engine.py — RAG 核心引擎
技术栈：ChromaDB + LangChain + 千问 Embedding
职责：文档解析 → 分块 → 向量化存储 → 检索
"""

import os
import logging
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings import DashScopeEmbeddings

logger = logging.getLogger("rag_engine")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md"}


class RAGEngine:

    def __init__(self):
        self._embeddings = None  # 懒加载，不在这里读 Key

        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        logger.info("RAGEngine 初始化完成，向量库路径：%s", CHROMA_DIR)

    @property
    def embeddings(self):
        """首次调用时才读取 Key，此时环境变量已由 __main__ 写入"""
        if self._embeddings is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError("未设置 DASHSCOPE_API_KEY，RAG 向量化不可用")
            self._embeddings = DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=api_key,
            )
        return self._embeddings

    def _load_document(self, file_path: str) -> List[str]:
        ext = Path(file_path).suffix.lower()
        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型：{ext}，仅支持 {sorted(ALLOWED_EXTENSIONS)}")

        docs = loader.load()
        full_text = "\n\n".join(doc.page_content for doc in docs)
        chunks = self.splitter.split_text(full_text)
        logger.info("文档解析完成：%s，共 %d 个文本块", file_path, len(chunks))
        return chunks

    def add_document(self, file_path: str, filename: str) -> Dict:
        chunks = self._load_document(file_path)
        if not chunks:
            raise ValueError(f"文档内容为空：{filename}")

        embeddings_list = self.embeddings.embed_documents(chunks)
        ids = [f"{filename}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

        existing = self.collection.get(where={"filename": filename})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            logger.info("删除旧向量：%s（%d 块）", filename, len(existing["ids"]))

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )
        logger.info("向量化存储完成：%s，%d 块", filename, len(chunks))
        return {"filename": filename, "chunks": len(chunks), "status": "ok"}

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": doc,
                "filename": meta.get("filename", "unknown"),
                "score": round(1 - dist, 4),
            })

        logger.info("检索完成：query='%s'，返回 %d 条", query[:30], len(retrieved))
        return retrieved

    def list_documents(self) -> List[str]:
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        filenames = sorted(set(m["filename"] for m in all_meta))
        return filenames


rag_engine = RAGEngine()