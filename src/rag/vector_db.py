import json
import os
import shutil
import inspect
import time
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

RAG_MANIFEST_NAME = ".rag_manifest.json"
RAG_SCHEMA_VERSION = 2


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
    schema_version: int,
) -> bool:
    """Manifest phải khớp để đảm bảo index đúng schema metadata hiện tại."""
    if manifest is None:
        return False
    return (
        manifest.get("embedding_model") == embedding_model
        and manifest.get("data_dir") == data_dir_resolved
        and manifest.get("collection_name") == collection_name
        and manifest.get("schema_version") == schema_version
    )


def clear_vector_store_dir(persist_dir: str) -> None:
    if os.path.isdir(persist_dir):
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                shutil.rmtree(persist_dir)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.3 * (2**attempt))
        raise PermissionError(
            f"Không thể xóa thư mục vector store '{persist_dir}' vì file đang bị tiến trình khác sử dụng "
            "(thường là chroma.sqlite3 trên Windows). "
            "Hãy đóng mọi chương trình đang mở file/thư mục này (ví dụ: SQLite viewer, terminal Python khác) rồi chạy lại."
        ) from last_err


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

    def _doc_matches_filters(self, metadata: dict, filters: dict) -> bool:
        for k, v in (filters or {}).items():
            if metadata.get(k) != v:
                return False
        return True

    def _similarity_search_with_score(self, query: str, *, k: int, filters: dict | None):
        fn = getattr(self.db, "similarity_search_with_score", None)
        if fn is None:
            raise AttributeError("Vector store does not support similarity_search_with_score")

        sig = inspect.signature(fn)
        kwargs = {}
        if "k" in sig.parameters:
            kwargs["k"] = k

        filters_supported = False
        if filters:
            if "filter" in sig.parameters:
                kwargs["filter"] = filters
                filters_supported = True
            elif "where" in sig.parameters:
                kwargs["where"] = filters
                filters_supported = True

        results = fn(query, **kwargs)
        if not filters or filters_supported:
            return results

        # Fallback: filter thủ công khi backend không hỗ trợ where/filter
        filtered = []
        for doc, score in results:
            meta = dict(getattr(doc, "metadata", {}) or {})
            if self._doc_matches_filters(meta, filters):
                filtered.append((doc, score))
        return filtered

    def _mmr_search(self, query: str, *, k: int, fetch_k: int, lambda_mult: float, filters: dict | None):
        fn = getattr(self.db, "max_marginal_relevance_search", None)
        if fn is None:
            raise AttributeError("Vector store does not support max_marginal_relevance_search")

        sig = inspect.signature(fn)
        kwargs = {}
        if "k" in sig.parameters:
            kwargs["k"] = k
        if "fetch_k" in sig.parameters:
            kwargs["fetch_k"] = fetch_k
        if "lambda_mult" in sig.parameters:
            kwargs["lambda_mult"] = lambda_mult

        if filters:
            if "filter" in sig.parameters:
                kwargs["filter"] = filters
            elif "where" in sig.parameters:
                kwargs["where"] = filters

        return fn(query, **kwargs)

    def search_locations(
        self,
        query: str,
        *,
        k: int = 10,
        filters: dict | None = None,
        strategy: str = "similarity",
        mmr_fetch_k: int = 50,
        mmr_lambda_mult: float = 0.5,
        rerank_enable: bool = False,
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_n: int = 50,
        dedupe_by_page: bool = True,
        max_chunks_per_page: int = 1,
        score_threshold: float | None = None,
        include_score: bool = False,
    ) -> list[dict]:
        """
        Trả về danh sách kết quả ưu tiên định vị theo (course, lecture, slide, page).

        Mặc định `max_chunks_per_page=1` để mỗi trang xuất hiện một lần trong top-k,
        phù hợp mục tiêu “đúng vị trí” hơn là “nhiều đoạn trùng trang”.
        """
        strategy_norm = (strategy or "similarity").strip().lower()
        if strategy_norm == "mmr":
            docs = self._mmr_search(
                query,
                k=max(k, 1),
                fetch_k=max(mmr_fetch_k, k),
                lambda_mult=mmr_lambda_mult,
                filters=filters,
            )
            raw = [(d, 0.0) for d in docs]  # MMR không có score
        else:
            fetch_k = max(k * 5, k) if filters else k
            raw = self._similarity_search_with_score(query, k=fetch_k, filters=filters)

        def normalize_hit(doc, score: float) -> dict:
            meta = dict(getattr(doc, "metadata", {}) or {})

            course = meta.get("course_name") or meta.get("course") or ""
            lecture = meta.get("lecture_name") or ""

            slide = meta.get("slide_file")
            if not slide:
                src = meta.get("slide_path") or meta.get("source") or ""
                slide = Path(src).name if src else ""

            page = meta.get("page_number")
            if page is None:
                try:
                    page = int(meta.get("page", 0)) + 1
                except Exception:
                    page = 1

            hit = {
                "course": course,
                "lecture": lecture,
                "slide": slide,
                "page": int(page),
                "content": getattr(doc, "page_content", "") or "",
            }
            if include_score:
                hit["score"] = float(score)
            return hit

        # Optional re-rank (Cross-Encoder): áp dụng trên tập ứng viên ban đầu
        if rerank_enable and raw:
            from src.rag.reranker import CrossEncoderReranker

            cand = raw[: max(1, min(rerank_top_n, len(raw)))]
            texts = [getattr(d, "page_content", "") or "" for d, _ in cand]
            rr = CrossEncoderReranker(rerank_model_name)
            order = rr.rerank(query, texts)
            raw = [cand[r.index] for r in order] + raw[len(cand) :]

        hits = []
        for doc, score in raw:
            try:
                s = float(score)
            except Exception:
                s = score

            if score_threshold is not None:
                # Chroma trả distance: nhỏ hơn là tốt hơn
                try:
                    if float(s) > float(score_threshold):
                        continue
                except Exception:
                    pass

            hits.append((doc, s))

        # similarity: distance nhỏ -> tốt; rerank đã sắp xếp raw nên chỉ sort khi chưa rerank
        if not rerank_enable and strategy_norm != "mmr":
            hits.sort(key=lambda x: x[1])

        if not dedupe_by_page:
            out = [normalize_hit(doc, score) for doc, score in hits[:k]]
            return out

        per_page_count: dict[str, int] = {}
        out: list[dict] = []
        for doc, score in hits:
            meta = dict(getattr(doc, "metadata", {}) or {})
            slide = meta.get("slide_file") or Path((meta.get("slide_path") or meta.get("source") or "")).name
            page = meta.get("page_number")
            if page is None:
                try:
                    page = int(meta.get("page", 0)) + 1
                except Exception:
                    page = 1
            key = f"{slide}::{int(page)}"

            n = per_page_count.get(key, 0)
            if n >= max_chunks_per_page:
                continue
            per_page_count[key] = n + 1

            out.append(normalize_hit(doc, score))
            if len(out) >= k:
                break

        return out