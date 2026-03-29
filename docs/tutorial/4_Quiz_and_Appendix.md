# AI VIET NAM – AI COURSE 2025

## Tutorial: RAG — Phần 4: Câu hỏi trắc nghiệm & Phụ lục

> **Tác giả:** Dương Trường Bình · Nguyễn Anh Khôi · Trần Đại Nhân · Dương Đình Thắng · Đinh Quang Vinh

---

## Mục lục

- [V. Câu hỏi trắc nghiệm](#v-câu-hỏi-trắc-nghiệm)
- [Phụ lục](#phụ-lục)

---

## V. Câu hỏi trắc nghiệm

**Câu 1.** Trong cơ chế Hybrid Search, tại sao chúng ta không thể cộng trực tiếp điểm số từ Vector Search và Keyword Search mà phải cần đến thuật toán Reciprocal Rank Fusion (RRF)?

- (a) Vì Vector Search trả về kết quả quá chậm so với Keyword Search nên cần RRF để đồng bộ hóa thời gian.
- **(b) Vì thang đo điểm số của hai phương pháp khác nhau, việc cộng trực tiếp sẽ khiến BM25 lấn át hoàn toàn Vector.**
- (c) Vì Vector Search chỉ tìm được từ đồng nghĩa, trong khi RRF giúp loại bỏ các từ đồng nghĩa sai.
- (d) Vì RRF là thuật toán duy nhất có thể chạy trên GPU để tăng tốc độ xử lý.

---

**Câu 2.** Quan sát hình minh họa dưới đây:

```
Văn bản gốc: "Mã truy cập bí mật là 2245. Vui lòng không chia sẻ."

Sau khi chunking (chunk_size = 6 từ):
  Chunk A: "Mã truy cập bí mật là"
  Chunk B: "2245. Vui lòng không chia sẻ."
```

Nếu người dùng hỏi: _"Mã truy cập bí mật là gì?"_, kết quả nào sau đây phản ánh chính xác nhất hoạt động của hệ thống RAG?

- (a) Hệ thống tìm thấy Chunk B vì nó chứa con số cụ thể, và LLM trả lời chính xác mã số.
- **(b) Hệ thống tìm thấy Chunk A, nhưng LLM sẽ không trả lời được mã số vì thông tin này nằm ở Chunk B, vốn không được tìm thấy.**
- (c) Hệ thống sẽ tự động tìm thấy cả Chunk A và Chunk B nhờ cơ chế suy luận bắc cầu của Vector Database.
- (d) Hệ thống không tìm thấy cả hai chunk vì câu văn bị cắt đôi làm hỏng cấu trúc ngữ pháp.

---

**Câu 3.** Tại sao mô hình Cross-Encoder lại có độ chính xác cao hơn Bi-Encoder, mặc dù Cross-Encoder chậm hơn rất nhiều?

- **(a) Vì Cross-Encoder quan sát được sự tương tác trực tiếp giữa từng token của câu hỏi và từng token của tài liệu cùng lúc.**
- (b) Vì Cross-Encoder nén câu hỏi và tài liệu thành các vector có số chiều lớn hơn.
- (c) Vì Cross-Encoder sử dụng cơ sở dữ liệu đồ thị Graph Database thay vì Vector Database.
- (d) Vì Cross-Encoder thực hiện việc tìm kiếm từ khóa chính xác thay vì tìm kiếm ngữ nghĩa.

---

**Câu 4.** Để khắc phục hiện tượng "Lost in the Middle", chiến lược sắp xếp ngữ cảnh nào sau đây là tối ưu nhất trước khi gửi cho LLM?

- (a) Sắp xếp theo thứ tự độ tương đồng giảm dần.
- (b) Sắp xếp ngẫu nhiên để tránh việc mô hình bị thiên kiến vào vị trí.
- **(c) Sắp xếp theo hình chữ U: Đưa các tài liệu quan trọng nhất vào đầu và cuối prompt, giấu các tài liệu ít quan trọng hơn vào giữa.**
- (d) Sắp xếp theo trình tự thời gian: Các tài liệu mới nhất được đưa lên đầu, cũ nhất xuống cuối.

---

**Câu 5.** Trong LangChain, "đơn vị cơ bản" để biểu diễn thông tin là gì?

- (a) Chunk.
- (b) Vector.
- (c) PromptTemplate.
- **(d) Document.**

---

**Câu 6.** Trong Ví dụ 1 ở phần **Loading Documents**, `PyPDFLoader` tạo ra Document theo đơn vị nào?

- (a) Mỗi file PDF là một Document.
- **(b) Mỗi trang trong file PDF là một Document.**
- (c) Mỗi chunk là một Document.
- (d) Mỗi câu là một Document.

---

**Câu 7.** Đối tượng nào biến các chunk thành vector embedding để lưu/tra cứu trong VectorStore?

```python
docs = PyPDFLoader("sample.pdf").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

emb = OpenAIEmbeddings(model="text-embedding-3-large")
vs = Chroma.from_documents(chunks, embedding=emb)

retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```

- (a) `PyPDFLoader`.
- (b) `RecursiveCharacterTextSplitter`.
- **(c) `OpenAIEmbeddings`.**
- (d) `Chroma` (Vector Store).

---

**Câu 8.** Quan sát đoạn code cấu hình Text Splitter dưới đây:

```python
class TextSplitter:
    def __init__(self, chunk_size=400, chunk_overlap=120):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
```

Trong bối cảnh xây dựng RAG, vai trò cụ thể của tham số `chunk_overlap=120` trong đoạn code trên là gì?

- (a) Chia văn bản thành các đoạn nhỏ có kích thước cố định đúng bằng 120 ký tự bất kể cấu trúc câu.
- (b) Tự động loại bỏ 120 ký tự đặc biệt hoặc khoảng trắng thừa ở đầu mỗi đoạn văn.
- **(c) Duy trì sự liên tục về ngữ cảnh giữa các đoạn liền kề bằng cách lặp lại 120 ký tự cuối của chunk trước sang chunk sau.**
- (d) Giới hạn độ dài tối đa của mỗi chunk là 120 ký tự để giảm tải bộ nhớ khi embedding.

---

**Câu 9.** Khi khởi tạo pipeline sinh văn bản cho hệ thống RAG, đoạn code cấu hình thường được thiết lập như sau:

```python
model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.2,   # Lưu ý tham số này
    max_new_tokens=450,
    do_sample=True,
    top_p=0.75
)
```

Trong ngữ cảnh của bài toán RAG, tại sao chiến lược đặt `temperature` ở mức thấp lại là tiêu chuẩn được khuyến nghị thay vì đặt ở mức cao?

- (a) Để ngăn chặn mô hình sinh ra các vòng lặp vô tận khi gặp các đoạn văn bản khó.
- (b) Để tăng tính đa dạng cho câu trả lời, giúp mô hình có thể diễn đạt cùng một ý theo nhiều văn phong khác nhau trong mỗi lần chạy.
- **(c) Để giảm thiểu sự sáng tạo ngẫu nhiên, buộc mô hình tập trung vào các từ có xác suất cao nhất nhằm đảm bảo câu trả lời bám sát vào tài liệu được cung cấp.**
- (d) Để tăng tốc độ xử lý của mô hình, vì giá trị temperature cao đòi hỏi nhiều tài nguyên tính toán hơn.

---

**Câu 10.** Trước khi đưa các tài liệu tìm được vào Prompt, hàm helper `format_docs` thực hiện một bước lọc dữ liệu như sau:

```python
def format_docs(docs):
    formatted = []
    seen = set()
    for doc in docs:
        content = doc.page_content.strip()
        if content and len(content) > 40 and content not in seen:
            formatted.append(content)
            seen.add(content)
    return "\n\n".join(formatted)
```

Mục đích thực tế của điều kiện `len(content) > 40` và `content not in seen` trong đoạn code trên là gì?

- (a) Đảm bảo chỉ có tối đa 40 tài liệu quan trọng nhất được gửi vào mô hình để xử lý.
- **(b) Loại bỏ các đoạn văn bản quá ngắn, thiếu ngữ nghĩa và ngăn chặn việc lặp lại thông tin để tối ưu hóa chất lượng Prompt.**
- (c) Kiểm tra xem tài liệu có chứa ít nhất 40 từ khóa quan trọng liên quan đến câu hỏi hay không.
- (d) Giới hạn tổng độ dài của prompt không vượt quá 40 token để tiết kiệm chi phí API và tránh lỗi Context Window.

---

## Phụ lục

1. **Solution:** File code cài đặt thực nghiệm có thể được tải tại đây.
2. **Q&A:** Bạn có thể đặt thêm câu hỏi về nội dung bài đọc trong group Facebook hỏi đáp. Tất cả câu hỏi sẽ được trả lời trong vòng tối đa 4 tiếng.
