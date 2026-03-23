import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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