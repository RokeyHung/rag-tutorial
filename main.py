import os
from dotenv import load_dotenv

from src.base.loader import SimpleLoader
from src.base.splitter import TextSplitter
from src.rag.vector_db import VectorDB
from src.rag.llm import get_hf_llm
from src.rag.rag_pipeline import OfflineRAG


load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "data_source/generative_ai")


def main():
    print("🚀 Loading documents...")
    loader = SimpleLoader()
    docs = loader.load_dir(DATA_DIR)

    print("✂️ Splitting...")
    splitter = TextSplitter()
    chunks = splitter.split(docs)

    print("🧠 Building vector DB...")
    vectordb = VectorDB(documents=chunks)
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