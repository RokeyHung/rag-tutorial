from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 120,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, documents):
        return self.split_per_page(documents)

    def split_per_page(self, documents):
        """
        Split theo từng Document đầu vào (mặc định: mỗi Document là 1 trang PDF),
        để bảo đảm mọi chunk đều nằm trong cùng một trang và giữ metadata trang.
        """
        chunks = []
        for doc in documents:
            sub = self.splitter.split_documents([doc])
            meta0 = dict(getattr(doc, "metadata", {}) or {})
            slide_file = meta0.get("slide_file", "")
            page_number = meta0.get("page_number")
            try:
                page_number_int = int(page_number) if page_number is not None else None
            except Exception:
                page_number_int = None
            location_key = (
                f"{slide_file}::{page_number_int}" if slide_file and page_number_int else None
            )

            for i, c in enumerate(sub):
                meta = dict(getattr(c, "metadata", {}) or {})
                meta["chunk_index"] = i
                if location_key is not None:
                    meta["location_key"] = location_key
                    meta["chunk_id"] = f"{location_key}::c{i}"
                c.metadata = meta
            chunks.extend(sub)
        return chunks