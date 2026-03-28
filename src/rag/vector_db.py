import json
import os
import shutil
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

RAG_MANIFEST_NAME = ".rag_manifest.json"


def chroma_collection_document_count(persist_dir: str, collection_name: str) -> int:
    """Số chunk/document trong collection đã persist (0 nếu chưa có)."""
    if not os.path.isdir(persist_dir):
        return 0
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        col = client.get_collection(name=collection_name)
        return int(col.count())
    except Exception:
        return 0


def rag_manifest_path(persist_dir: str) -> Path:
    return Path(persist_dir) / RAG_MANIFEST_NAME


def read_rag_manifest(persist_dir: str) -> dict | None:
    path = rag_manifest_path(persist_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_rag_manifest(persist_dir: str, payload: dict) -> None:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    rag_manifest_path(persist_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rag_manifest_matches(
    manifest: dict | None,
    *,
    embedding_model: str,
    data_dir_resolved: str,
    collection_name: str,
) -> bool:
    """Index cũ không có manifest vẫn được coi là dùng được (tương thích ngược)."""
    if manifest is None:
        return True
    return (
        manifest.get("embedding_model") == embedding_model
        and manifest.get("data_dir") == data_dir_resolved
        and manifest.get("collection_name") == collection_name
    )


def clear_vector_store_dir(persist_dir: str) -> None:
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)


class VectorDB:
    def __init__(
        self,
        documents=None,
        embedding_model: str | None = None,
        collection_name: str | None = None,
        persist_dir: str | None = None,
    ):
        embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "vietnamese_docs")

        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        if documents is None or len(documents) == 0:
            db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
            )
        else:
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
        return db

    def get_retriever(self, search_kwargs: dict = None):
        if search_kwargs is None:
            search_kwargs = {"k": 4}

        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )