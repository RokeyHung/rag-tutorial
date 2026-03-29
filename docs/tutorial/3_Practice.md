# AI VIET NAM – AI COURSE 2025

## Tutorial: RAG — Phần 3: Thực hành

> **Tác giả:** Dương Trường Bình · Nguyễn Anh Khôi · Trần Đại Nhân · Dương Đình Thắng · Đinh Quang Vinh

---

## Mục lục

- [IV.1. Chuẩn bị môi trường và cấu trúc dự án](#iv1-chuẩn-bị-môi-trường-và-cấu-trúc-dự-án)
- [IV.2. Chuẩn bị dữ liệu PDF đầu vào](#iv2-chuẩn-bị-dữ-liệu-pdf-đầu-vào)
- [IV.3. Tiền xử lý văn bản và Chunking](#iv3-tiền-xử-lý-văn-bản-và-chunking)
- [IV.4. Xây dựng Vector Database](#iv4-xây-dựng-vector-database)
- [IV.5. Khởi tạo LLM và xây dựng RAG Chain](#iv5-khởi-tạo-llm-và-xây-dựng-rag-chain)
- [IV.6. Xây dựng giao diện ứng dụng](#iv6-xây-dựng-giao-diện-ứng-dụng)

---

## IV. Thực hành

Trong phần này, chúng ta xây dựng một **Question Answering System** (hệ thống hỏi đáp) dành cho các tài liệu PDF tiếng Việt.

**Kiến trúc hệ thống:**

```
AIVN Documents
      ↓
  [Chunking]
      ↓
[Embedding Model]
      ↓
 [Vector Store: ChromaDB]
      ↓
[Similarity Search] ← Query Vector ← [Embedding Model] ← User's Query
      ↓
  [Top-K Context]
      ↓
[Prompt Template]
      ↓
  [Qwen Model]
      ↓
    Answer → User
```

> _Hình 10: Kiến trúc hệ thống RAG sử dụng ChromaDB và mô hình ngôn ngữ Qwen._

**Môi trường làm việc:**

- **Platform:** Google Colab với GPU T4
- **Python:** 3.12
- **Dữ liệu:** Các bài viết public của AI VIET NAM
- **LLM:** Qwen2.5-3B-Instruct
- **Vector DB:** ChromaDB
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

---

### IV.1. Chuẩn bị môi trường và cấu trúc dự án

**Cài đặt các thư viện:**

```python
# Cài đặt các thư viện chính cho mô hình ngôn ngữ, embedding, RAG và giao diện web
!pip install -q \
    "torch>=2.0.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "huggingface-hub>=0.23.0" \
    "sentence-transformers>=2.7.0" \
    "langchain>=0.2.0" \
    "langchain-core>=0.2.0" \
    "langchain-community>=0.1.0" \
    "langchain-text-splitters>=0.2.0" \
    "chromadb>=0.5.0" \
    "langchain-chroma>=0.2.0" \
    "pypdf>=4.2.0" \
    "langserve[all]>=0.1.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.30.0" \
    "gradio>=5.0.0" \
    "langchain-huggingface" \
    "wget"
```

**Thiết lập cấu trúc dự án:**

```python
import os
import sys

PROJECT_ROOT = "/content/rag_langchain"

# Đặt token để tải mô hình private, hãy thay bằng token của bạn
os.environ["HF_TOKEN"] = "YOUR_HUGGINGFACE_TOKEN"

# Tạo thư mục cho dữ liệu và mã nguồn
os.makedirs(os.path.join(PROJECT_ROOT, "data_source", "generative_ai"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "src", "base"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "src", "rag"), exist_ok=True)

os.chdir(PROJECT_ROOT)

# Thêm PROJECT_ROOT vào sys.path để có thể import các module trong src
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**Tạo các file `__init__.py`** để các thư mục mã nguồn có thể được sử dụng như các package Python:

```bash
%%bash
touch /content/rag_langchain/src/__init__.py
touch /content/rag_langchain/src/base/__init__.py
touch /content/rag_langchain/src/rag/__init__.py
```

**Cấu trúc thư mục sau khi thiết lập:**

```
/content/rag_langchain/
├── data_source/
│   └── generative_ai/       # Chứa các file PDF
├── src/
│   ├── __init__.py
│   ├── base/
│   │   └── __init__.py
│   └── rag/
│       └── __init__.py
```

---

### IV.2. Chuẩn bị dữ liệu PDF đầu vào

Tải các tài liệu PDF trích từ các bài viết công khai của AI VIET NAM:

```python
import os
import wget

# Thư mục lưu các tài liệu PDF
DATA_DIR = "/content/rag_langchain/data_source/generative_ai"
os.makedirs(DATA_DIR, exist_ok=True)

# Danh sách các tài liệu PDF mẫu sẽ dùng làm dữ liệu cho hệ thống RAG
pdf_links = [
    {
        "title": "Vòng lặp for và ứng dụng",
        "url": "https://docs.google.com/uc?export=download&id=1_zJnj5qORwzMH6vgftGzkYu4fFew6Pq4"
    },
    {
        "title": "Giám sát hệ thống AI với Grafana và Prometheus",
        "url": "https://docs.google.com/uc?export=download&id=1gZWLJddiuLd-ZZ8j_Nmfu_r7fISGRW6J"
    },
    {
        "title": "Các Thước Đo Đánh Giá Mô Hình Hồi Quy",
        "url": "https://docs.google.com/uc?export=download&id=1C-f9pNW0mkMxaakDcpliN3isTVqRQtR3"
    },
    {
        "title": "A simple, strong baseline for Long-Term Forecasts",
        "url": "https://docs.google.com/uc?export=download&id=16KFeWi0ONqV3ZJYAgC_20y7bxUNGd9hU"
    },
]

# Tải các file PDF về thư mục dữ liệu nếu chưa tồn tại
for pdf_info in pdf_links:
    save_path = os.path.join(DATA_DIR, f"{pdf_info['title']}.pdf")
    if not os.path.exists(save_path):
        try:
            wget.download(pdf_info["url"], out=save_path)
        except Exception as e:
            pass
```

---

### IV.3. Tiền xử lý văn bản và Chunking

**Làm sạch văn bản tiếng Việt**

Văn bản trích xuất từ PDF thường chứa ký tự đặc biệt, khoảng trắng thừa, hoặc các vấn đề encoding. Hàm `clean_vietnamese_text` chuẩn hóa văn bản tiếng Việt:

```python
import re
import unicodedata
from typing import List

def clean_vietnamese_text(text: str) -> str:
    # Chuẩn hóa Unicode về dạng NFC cho tiếng Việt
    text = unicodedata.normalize('NFC', text)
    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith('C') or char in '\n\t'
    )

    # Gộp khoảng trắng thừa và dòng trống
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()
```

**Tải và chia nhỏ tài liệu PDF**

Xây dựng hai class chính: `SimpleLoader` để tải tài liệu PDF và `TextSplitter` để chia văn bản thành các chunk:

```python
import glob
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SimpleLoader:
    def load_pdf(self, pdf_file: str):
        """Tải một file PDF và làm sạch nội dung văn bản."""
        docs = PyPDFLoader(pdf_file, extract_images=True).load()
        for doc in docs:
            doc.page_content = clean_vietnamese_text(doc.page_content)
        return docs

    def load_dir(self, dir_path: str) -> List:
        """Tải tất cả file PDF trong một thư mục."""
        pdf_files = glob.glob(f"{dir_path}/*.pdf")
        if not pdf_files:
            raise ValueError(f"No PDF files found in {dir_path}")

        all_docs = []
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                all_docs.extend(self.load_pdf(pdf_file))
            except Exception as e:
                pass
        return all_docs


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 120,
    ):
        """Khởi tạo với chunk_size và chunk_overlap phù hợp cho tiếng Việt."""
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, documents):
        """Chia danh sách documents thành các chunk nhỏ hơn."""
        return self.splitter.split_documents(documents)
```

---

### IV.4. Xây dựng Vector Database

Các chunk đã được chia nhỏ sẽ được chuyển thành vector qua mô hình embedding đa ngôn ngữ từ HuggingFace, sau đó lưu vào **Chroma**. Class `VectorDB` quản lý toàn bộ quy trình từ khởi tạo embedding model, index document chunks, đến cung cấp retriever.

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(
        self,
        documents=None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "vietnamese_docs",
        persist_dir: str = "/content/chroma_data",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        """Xây dựng hoặc tải Vector Database."""
        # Trường hợp tải database đã có từ persist_dir
        if documents is None or len(documents) == 0:
            db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
            )
        # Trường hợp tạo mới database từ documents
        else:
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
        return db

    def get_retriever(self, search_kwargs: dict = None):
        """Tạo retriever để tìm kiếm tương đồng."""
        if search_kwargs is None:
            search_kwargs = {"k": 4}  # Mặc định lấy 4 document gần nhất

        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )
```

---

### IV.5. Khởi tạo LLM và xây dựng RAG Chain

#### Khởi tạo LLM

Hàm `get_hf_llm` khởi tạo mô hình ngôn ngữ từ HuggingFace và đóng gói thành LangChain pipeline. Ở đây, ta sử dụng **Qwen2.5-3B-Instruct** — một mô hình instruction-tuned hiệu quả cho tác vụ hỏi đáp và sinh văn bản.

```python
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline

def get_hf_llm(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    temperature: float = 0.2,
    max_new_tokens: int = 450,
    **kwargs
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tạo text generation pipeline
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.75
    )

    # Đóng gói pipeline thành LangChain LLM
    llm = HuggingFacePipeline(pipeline=model_pipeline, model_kwargs=kwargs)
    return llm
```

> **Lưu ý về `temperature=0.2`:** Trong RAG, temperature thấp được khuyến nghị để **giảm sáng tạo ngẫu nhiên**, buộc mô hình tập trung vào các từ có xác suất cao nhất và bám sát tài liệu được cung cấp, tránh hallucination.

#### Định nghĩa RAG Chain và Parser

`FocusedAnswerParser` làm sạch và rút gọn câu trả lời. `OfflineRAG` kết nối retriever, prompt template và LLM thành một chain hoàn chỉnh.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class FocusedAnswerParser(StrOutputParser):
    def parse(self, text: str) -> str:
        text = text.strip()
        if "[TRẢ LỜI]:" in text:
            answer = text.split("[TRẢ LỜI]:")[-1].strip()
        else:
            answer = text

        answer = re.sub(r'^\s*[\\-\*]\s*', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'\n+', ' ', answer)
        lines = [
            line.strip()
            for line in answer.split('. ')
            if line.strip() and len(line.strip()) > 5
        ]
        return '. '.join(lines[:5]) + ('.' if lines else '')


class OfflineRAG:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template("""
Bạn là trợ lý AI phân tích tài liệu tiếng Việt.

[TÀI LIỆU]:
{context}

[CÂU HỎI]:
{question}

Hãy trả lời dựa trên tài liệu. Nếu tài liệu không có thông tin, nói rõ "Không có thông tin".
Trả lời đầy đủ thông tin (3-5 câu chi tiết), không thêm bất kỳ chi tiết nào ngoài tài liệu.
[TRẢ LỜI]:""")

        self.answer_parser = FocusedAnswerParser()

    def get_chain(self, retriever):
        """Xây dựng RAG chain từ retriever."""
        def format_docs(docs):
            formatted = []
            seen = set()
            for doc in docs:
                content = doc.page_content.strip()
                if content and len(content) > 40 and content not in seen:
                    formatted.append(content)
                    seen.add(content)
            return "\n\n".join(formatted)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.answer_parser
        )
        return rag_chain
```

#### Kết nối toàn bộ pipeline

```python
os.chdir("/content/rag_langchain")

# Khởi tạo LLM
llm = get_hf_llm()

data_dir = "/content/rag_langchain/data_source/generative_ai"

# Load và xử lý documents
loader = SimpleLoader()
text_splitter = TextSplitter(chunk_size=400, chunk_overlap=120)

raw_docs = loader.load_dir(data_dir)
split_docs = text_splitter.split(raw_docs)

# Xây dựng Vector Database và Retriever
vdb = VectorDB(documents=split_docs)
retriever = vdb.get_retriever(search_kwargs={"k": 4})

# Tạo RAG Chain
rag = OfflineRAG(llm)
rag_chain = rag.get_chain(retriever)

# Hàm xử lý câu hỏi
def answer_question(question: str) -> str:
    try:
        return rag_chain.invoke(question)
    except Exception as e:
        return f"Error: {str(e)}"
```

---

### IV.6. Xây dựng giao diện ứng dụng

Xây dựng giao diện đơn giản sử dụng **Gradio** để người dùng tương tác trực tiếp với hệ thống RAG:

```python
import gradio as gr

# Xây dựng giao diện với Gradio Blocks
with gr.Blocks(title="RAG Vietnamese QA") as demo:
    gr.Markdown("# RAG - Hỏi Đáp về Tài Liệu")

    with gr.Row():
        # Cột bên trái: Input câu hỏi
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Câu hỏi",
                placeholder="Ví dụ: Vì sao classification lại không thể chỉ nhìn accuracy để đánh giá?",
                lines=3,
            )
            submit_btn = gr.Button("Gửi", variant="primary")

        # Cột bên phải: Output câu trả lời
        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="Câu trả lời",
                lines=6,
                interactive=False
            )

    # Kết nối button với hàm xử lý
    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=answer_output,
    )

demo.launch(share=True)
```

Khi chạy cell này, một đường link Gradio sẽ được tạo ra. Người dùng có thể nhập câu hỏi liên quan đến các tài liệu đã được index và quan sát cách hệ thống RAG truy xuất ngữ cảnh và sinh câu trả lời.
