# AI VIET NAM – AI COURSE 2025

## Tutorial: RAG — Phần 2: Thư viện LangChain

> **Tác giả:** Dương Trường Bình · Nguyễn Anh Khôi · Trần Đại Nhân · Dương Đình Thắng · Đinh Quang Vinh

---

## Mục lục

- [III.1. Giới thiệu](#iii1-giới-thiệu)
- [III.2. Các thành phần cốt lõi](#iii2-các-thành-phần-cốt-lõi)
  - [1. Documents và Document Loaders](#1-documents-và-document-loaders)
  - [2. Embeddings](#2-embeddings)
  - [3. Vector Stores](#3-vector-stores)
  - [4. Retrievers](#4-retrievers)

---

## III. Thư viện LangChain

### III.1. Giới thiệu

**LangChain** là một framework mã nguồn mở mạnh mẽ, được thiết kế để đơn giản hóa quá trình phát triển các ứng dụng sử dụng mô hình ngôn ngữ lớn (LLM). LangChain đóng vai trò liên kết các thành phần dữ liệu, xử lý logic và mô hình lại với nhau.

Trong bài toán RAG, LangChain giải quyết vấn đề cốt lõi là sự rời rạc giữa nguồn dữ liệu và khả năng của LLM. Thư viện này hỗ trợ toàn bộ vòng đời của ứng dụng RAG mà không cần người lập trình phải tự xây dựng các kết nối thủ công phức tạp.

**Tổng quan pipeline trong LangChain:**

```
Documents
    ↓ [Document Loaders]
Document objects (page_content + metadata + id)
    ↓ [Text Splitter]
Chunks (Document objects nhỏ hơn)
    ↓ [Embedding Model]
Vectors
    ↓ [Vector Store]
Indexed Vector DB
    ↓ [Retriever (k=1)]
Most relevant chunk(s) cho Query
```

---

### III.2. Các thành phần cốt lõi

#### 1. Documents và Document Loaders

Trong LangChain, đơn vị cơ bản để biểu diễn thông tin là đối tượng **`Document`**.

**Cấu trúc `Document`:**

| Thuộc tính     | Kiểu dữ liệu | Mô tả                                                             |
| -------------- | ------------ | ----------------------------------------------------------------- |
| `page_content` | `str`        | Chuỗi ký tự chứa nội dung văn bản                                 |
| `metadata`     | `dict`       | Thông tin bổ sung tùy ý (nguồn gốc, số trang, ngày xuất bản, ...) |
| `id`           | `str`        | Mã định danh chuỗi cho tài liệu                                   |

**Loading Documents:**

LangChain cung cấp hệ sinh thái Document Loaders tích hợp với hàng trăm nguồn dữ liệu khác nhau (PDF, CSV, HTML, ...). Ví dụ, sử dụng `PyPDFLoader` để nạp một tài liệu PDF:

```python
from langchain_community.document_loaders import PyPDFLoader

file_path = "../ai-vietnam-2025.pdf"
loader = PyPDFLoader(file_path)
doc = loader.load()
```

> **Lưu ý:** `PyPDFLoader` tạo ra mỗi **trang PDF** thành một đối tượng `Document` riêng biệt.

**Splitting:**

Một trang tài liệu gốc thường quá dài hoặc chứa nhiều thông tin hỗn tạp. Công cụ **Text Splitters** chia tài liệu thành các "chunk" nhỏ hơn. Phương pháp phổ biến là `RecursiveCharacterTextSplitter`:

- Phân chia đệ quy dựa trên các ký tự phân cách (`\n\n`, `\n`, khoảng trắng) để giữ nguyên vẹn ngữ nghĩa.
- Giữ lại một phần ký tự trùng lặp giữa các đoạn liền kề để đảm bảo ngữ cảnh không bị ngắt quãng.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Kích thước của mỗi chunk
    chunk_overlap=200,  # Giữ 200 ký tự của đoạn liền kề
)
all_splits = text_splitter.split_documents(doc)
```

---

#### 2. Embeddings

**Vector Search** là phương pháp phổ biến để lưu trữ và tìm kiếm trên dữ liệu phi cấu trúc.

**Nguyên lý hoạt động:**

- Mô hình Embedding chuyển đổi một văn bản thành một **vector số thực** biểu diễn cho văn bản đó.
- Những văn bản có ý nghĩa tương tự sẽ có **vector gần nhau** trong không gian hình học.
- Các chỉ số như **cosine similarity** được dùng để đo độ tương đồng giữa các văn bản.

LangChain hỗ trợ giao diện chuẩn cho nhiều nhà cung cấp mô hình embedding: OpenAI, Google, HuggingFace, ...

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Tạo vector cho một đoạn văn bản
vector_1 = embeddings.embed_query(all_splits[0].page_content)
print(len(vector_1))
# Kết quả: 1536 (số chiều của vector)
```

---

#### 3. Vector Stores

`VectorStore` trong LangChain chịu trách nhiệm:

- **Indexing:** Lưu trữ các đối tượng `Document` và vector tương ứng thông qua `add_documents`.
- **Querying:** Tìm kiếm văn bản dựa trên độ tương đồng vector thông qua `similarity_search`.

LangChain tích hợp với nhiều loại vector store:

| Loại                      | Ví dụ                            |
| ------------------------- | -------------------------------- |
| Lưu trong bộ nhớ          | `FAISS`, `InMemoryVectorStore`   |
| Cơ sở dữ liệu chuyên dụng | `Chroma`, `Pinecone`, `Postgres` |

```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

# Thêm documents vào vector store
ids = vector_store.add_documents(documents=all_splits)

# Tìm kiếm tương đồng
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
```

---

#### 4. Retrievers

Trong khi `VectorStore` là nơi lưu trữ, **Retriever** được dùng để truy vấn dữ liệu. Retrievers trong LangChain là các **`Runnable`**, cho phép kết nối dễ dàng vào các chuỗi xử lý (chains).

**`VectorStoreRetriever`**

Cách đơn giản nhất là tạo Retriever từ một VectorStore thông qua `.as_retriever()`:

```python
# Tạo Retriever từ Vector Store
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}  # Chỉ lấy 1 kết quả tốt nhất
)

# Thực thi tìm kiếm (Batch)
retriever.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?"
])
```

**Các loại `search_type`:**

| Giá trị                        | Mô tả                                                                  |
| ------------------------------ | ---------------------------------------------------------------------- |
| `"similarity"`                 | Tìm kiếm tương đồng mặc định                                           |
| `"mmr"`                        | Maximum Marginal Relevance — cân bằng giữa độ tương đồng và độ đa dạng |
| `"similarity_score_threshold"` | Lọc bỏ kết quả có điểm tương đồng thấp hơn ngưỡng quy định             |
