import os
import json
from pathlib import Path

from dotenv import load_dotenv

from src.base.env import env_bool, env_int, env_str
from src.base.loader import SimpleLoader, format_pdf_courses_report
from src.base.splitter import TextSplitter
from src.rag.vector_db import (
    RAG_SCHEMA_VERSION,
    VectorDB,
    chroma_collection_document_count,
    clear_vector_store_dir,
    rag_manifest_matches,
    read_rag_manifest,
    write_rag_manifest,
)


load_dotenv()
DATA_DIR = env_str("DATA_DIR", "data_source/pdf") or "data_source/pdf"


def main():
    persist_dir = env_str("CHROMA_PERSIST_DIR", "./chroma_data") or "./chroma_data"
    collection_name = env_str("CHROMA_COLLECTION_NAME", "vietnamese_docs") or "vietnamese_docs"
    embedding_model = env_str(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    data_dir_resolved = str(Path(DATA_DIR).resolve())
    rebuild = env_bool("REBUILD_VECTOR_DB", False)
    top_k = env_int("TOP_K", 5, min_value=1, max_value=50)
    use_llm = env_bool("RAG_USE_LLM", True)
    search_strategy = env_str("RAG_SEARCH_STRATEGY", "similarity") or "similarity"
    rerank_enable = env_bool("RAG_RERANK_ENABLE", False)
    rerank_model = env_str("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n = env_int("RAG_RERANK_TOP_N", 50, min_value=1, max_value=200)

    if rebuild:
        print("🧹 REBUILD_VECTOR_DB: xóa index cũ trong", persist_dir)
        clear_vector_store_dir(persist_dir)

    manifest = read_rag_manifest(persist_dir)
    count = chroma_collection_document_count(persist_dir, collection_name)
    reuse_index = (
        not rebuild
        and count > 0
        and rag_manifest_matches(
            manifest,
            embedding_model=embedding_model,
            data_dir_resolved=data_dir_resolved,
            collection_name=collection_name,
            schema_version=RAG_SCHEMA_VERSION,
        )
    )

    if reuse_index:
        print(f"📂 Dùng vector DB đã lưu ({count} chunk) — bỏ qua nạp PDF và embedding lại.")
        vectordb = VectorDB(documents=None)
    else:
        if not rebuild and count > 0:
            print("🧹 Cấu hình DATA_DIR / embedding / collection đổi — xây lại index.")
            clear_vector_store_dir(persist_dir)

        print("🚀 Loading documents...")
        loader = SimpleLoader()
        root = Path(DATA_DIR)
        if root.is_dir() and any(p.is_dir() for p in root.iterdir()):
            load_result = loader.load_pdf_courses(str(root))
            print(format_pdf_courses_report(load_result))
            docs = load_result.documents
        else:
            docs = loader.load_dir(DATA_DIR)

        print("✂️ Splitting...")
        splitter = TextSplitter()
        chunks = splitter.split(docs)

        print("🧠 Building vector DB...")
        vectordb = VectorDB(documents=chunks)
        write_rag_manifest(
            persist_dir,
            {
                "schema_version": RAG_SCHEMA_VERSION,
                "embedding_model": embedding_model,
                "data_dir": data_dir_resolved,
                "collection_name": collection_name,
            },
        )

    rag = None
    if use_llm:
        from src.rag.llm import get_hf_llm
        from src.rag.rag_pipeline import OfflineRAG

        retriever = vectordb.get_retriever()
        print("🤖 Loading LLM...")
        llm = get_hf_llm()
        print("🔗 Building RAG...")
        rag = OfflineRAG(retriever, llm)

    print("\n✅ Ready!")

    while True:
        query = input("\n❓ Question: ")
        if query == "exit":
            break

        hits = vectordb.search_locations(
            query,
            k=top_k,
            dedupe_by_page=True,
            strategy=search_strategy,
            rerank_enable=rerank_enable,
            rerank_model_name=rerank_model,
            rerank_top_n=rerank_top_n,
        )
        print("\n📌 Locations:")
        print(json.dumps(hits, ensure_ascii=False, indent=2))

        if rag is not None:
            answer = rag.ask(query)
            print("\n💡 Answer:", answer)


if __name__ == "__main__":
    main()