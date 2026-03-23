import re
import unicodedata
from typing import List
from langchain_community.document_loaders import PyPDFLoader


def clean_vietnamese_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)

    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("C") or char in "\n\t"
    )

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)

    return text.strip()


class SimpleLoader:
    def load_pdf(self, pdf_file: str):
        docs = PyPDFLoader(pdf_file).load()

        for doc in docs:
            doc.page_content = clean_vietnamese_text(doc.page_content)

        return docs

    def load_dir(self, dir_path: str) -> List:
        import glob
        from tqdm import tqdm

        pdf_files = glob.glob(f"{dir_path}/*.pdf")

        if not pdf_files:
            raise ValueError(f"No PDF files found in {dir_path}")

        all_docs = []

        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                all_docs.extend(self.load_pdf(pdf_file))
            except Exception:
                pass

        return all_docs