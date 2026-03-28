import os
from pathlib import Path

from dotenv import load_dotenv

from src.base.loader import SimpleLoader, format_pdf_courses_report
from src.base.splitter import TextSplitter
from src.rag.llm import get_hf_llm
from src.rag.rag_pipeline import OfflineRAG
from src.rag.vector_db import (
    VectorDB,
    chroma_collection_document_count,
    clear_vector_store_dir,
    rag_manifest_matches,
    read_rag_manifest,
    write_rag_manifest,
)


load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "data_source/pdf")


def main():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "vietnamese_docs")
    embedding_model = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    data_dir_resolved = str(Path(DATA_DIR).resolve())
    rebuild = os.getenv("REBUILD_VECTOR_DB", "").lower() in ("1", "true", "yes")

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
                "embedding_model": embedding_model,
                "data_dir": data_dir_resolved,
                "collection_name": collection_name,
            },
        )

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

        answer = rag.ask(query)
        print("\n💡 Answer:", answer)


if __name__ == "__main__":
    main()