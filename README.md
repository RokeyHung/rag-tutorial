# RAG Tutorial (PDF Slides) — Truy xuất chính xác theo Course/Lecture/Slide/Page

Project này xây dựng một pipeline RAG cho **PDF slide bài giảng**, với mục tiêu chính là **tìm đúng vị trí** (môn học → buổi học → file slide → số trang) và trả về kết quả dạng JSON có metadata đầy đủ.

## Tính năng chính

- **Ingest PDF theo trang**: mỗi trang PDF là một `Document`, sau đó split thành nhiều chunk **nhưng không vượt qua ranh giới trang**.
- **Metadata định vị** được gắn vào mọi chunk:
  - `course_name`
  - `lecture_name`
  - `lecture_id` (slug ổn định)
  - `slide_file`
  - `slide_path`
  - `page_number` (1-based)
  - `location_key`, `chunk_id`, `chunk_index` (phục vụ dedupe/citation/eval)
- **Vector DB (Chroma)**: lưu embedding + metadata, truy xuất trả về đúng `(slide, page)`.
- **Re-ranking (tuỳ chọn)**: chiến lược “Retrieve many → Re-rank few” bằng Cross-Encoder để tăng precision.
- **Citation cho LLM (tuỳ chọn)**: prompt yêu cầu trích dẫn theo nhãn `[course|lecture|slide|pX]`.
- **Đánh giá theo định vị**: tính `recall@k` và `mrr@k` dựa trên nhãn đúng `(slide, page)`.

## Cấu trúc dữ liệu đầu vào (khuyến nghị)

Đặt PDF theo cấu trúc:

```text
data_source/pdf/
  <course_name>/
    <lecture_name_or_id>/
      *.pdf
```

Ví dụ:

```text
data_source/pdf/
  MachineLearning/
    Week03-NeuralNetworks/
      L03_neural_networks.pdf
```

Hệ thống suy ra:

- `course_name` = tên thư mục cấp 1
- `lecture_name` = tên thư mục cấp 2 (nếu không có thì `default`)
- `slide_file` = tên file PDF
- `page_number` = số trang (1-based)

## Cài đặt

### 1) Tạo môi trường Python và cài dependency

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2) Cấu hình `.env`

Project dùng `python-dotenv` và đọc config từ `.env`.

Ví dụ `.env` (tham khảo):

```env
# Dữ liệu
DATA_DIR=data_source/pdf

# Chroma
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_COLLECTION_NAME=vietnamese_docs

# Index rebuild (Windows có thể bị lock chroma.sqlite3)
REBUILD_VECTOR_DB=0

# Retrieval
TOP_K=5
RAG_SEARCH_STRATEGY=similarity   # similarity | mmr

# Re-ranking (Cross-Encoder)
RAG_RERANK_ENABLE=false
RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_RERANK_TOP_N=50

# Embedding
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# LLM (tuỳ chọn)
RAG_USE_LLM=true
LLM_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
HF_TOKEN=your_hf_token_here
```

Lưu ý:

- **Không commit token** (ví dụ `HF_TOKEN`) lên git.
- Trên Windows, nếu `REBUILD_VECTOR_DB=1` mà file `chroma.sqlite3` đang bị tiến trình khác giữ, thao tác xóa sẽ fail. Hãy đóng SQLite viewer/terminal Python khác rồi chạy lại.

## Chạy demo CLI

Chạy chương trình chính:

```bash
python main.py
```

Khi nhập câu hỏi, chương trình sẽ in:

1. **Locations JSON** (ưu tiên định vị)
2. **Answer** (nếu `RAG_USE_LLM=true`)

Ví dụ output locations:

```json
[
  {
    "course": "MachineLearning",
    "lecture": "Week03-NeuralNetworks",
    "slide": "L03_neural_networks.pdf",
    "page": 12,
    "content": "…đoạn text liên quan…"
  }
]
```

## Cách tăng độ chính xác (precision)

### Bật re-rank (khuyến nghị khi truy vấn khó)

```env
RAG_RERANK_ENABLE=true
RAG_RERANK_TOP_N=50
```

Cross-Encoder sẽ đọc trực tiếp cặp (query, chunk) để sắp xếp lại top candidate, thường cải thiện tốt với các truy vấn có phủ định/quan hệ nguyên nhân-kết quả.

### MMR để tăng đa dạng kết quả

```env
RAG_SEARCH_STRATEGY=mmr
```

MMR giúp kết quả ít bị “dồn” vào một vùng nội dung gần nhau.

## Đánh giá theo định vị (Recall@k / MRR@k)

Tạo một file gold (JSON hoặc JSONL) gồm danh sách truy vấn và vị trí đúng theo `(slide, page)`.

Ví dụ `gold.json`:

```json
[
  {
    "query": "backpropagation là gì?",
    "gold": [{ "slide": "L03_neural_networks.pdf", "page": 12 }]
  }
]
```

Chạy đánh giá:

```bash
python -m src.rag.eval_locations --gold gold.json --k 5
```

Output:

```json
{
  "k": 5,
  "recall@k": 0.8,
  "mrr@k": 0.6
}
```

## File quan trọng trong codebase

- `main.py`: CLI demo (locations + LLM answer tuỳ chọn)
- `src/base/loader.py`: load PDF theo trang + gắn metadata `course/lecture/slide/page`
- `src/base/splitter.py`: split theo trang, gắn `location_key/chunk_id`
- `src/rag/vector_db.py`: Chroma + API `search_locations()` (dedupe theo trang, MMR, rerank)
- `src/rag/rag_pipeline.py`: prompt có citation + reorder context (U-shape)
- `src/rag/eval_locations.py`: đánh giá Recall@k / MRR@k theo định vị
- `src/rag/llm.py`: khởi tạo HuggingFace LLM pipeline
