# AI VIET NAM – AI COURSE 2025

## Tutorial: RAG — Phần 1: Dẫn nhập & Cơ sở lý thuyết

> **Tác giả:** Dương Trường Bình · Nguyễn Anh Khôi · Trần Đại Nhân · Dương Đình Thắng · Đinh Quang Vinh

---

## Mục lục

- [I. Dẫn nhập](#i-dẫn-nhập)
- [II. Cơ sở lý thuyết về RAG](#ii-cơ-sở-lý-thuyết-về-rag)
  - [II.1. Sự chuyển dịch sang In-Context RAG](#ii1-sự-chuyển-dịch-sang-in-context-rag)
  - [II.2. Kiến trúc RAG hiện đại](#ii2-kiến-trúc-rag-hiện-đại)
  - [II.3. Mở rộng: Phân tích vai trò của các thành phần](#ii3-mở-rộng-phân-tích-vai-trò-của-các-thành-phần)

---

## Bảng thuật ngữ

| Thuật ngữ               | Mô tả                                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Hallucination**       | Hiện tượng mô hình sinh ra thông tin sai lệch, bịa đặt hoặc không có thực nhưng với văn phong tự tin.                       |
| **Knowledge Cutoff**    | Mốc thời gian giới hạn của dữ liệu huấn luyện, khiến mô hình không biết các sự kiện xảy ra sau đó.                          |
| **Fine-tuning**         | Quá trình huấn luyện tiếp mô hình đã pre-trained trên tập dữ liệu chuyên biệt để cập nhật trọng số.                         |
| **In-Context Learning** | Khả năng LLM học và thực hiện tác vụ dựa trên ngữ cảnh hoặc ví dụ được cung cấp trong prompt mà không cần cập nhật tham số. |
| **Vector Embeddings**   | Biểu diễn dữ liệu (text, image) dưới dạng vector số thực trong không gian n-chiều.                                          |
| **Semantic Search**     | Tìm kiếm dựa trên sự tương đồng về ý nghĩa thay vì chỉ khớp từ khóa.                                                        |
| **Chunking**            | Kỹ thuật chia nhỏ văn bản dài thành các đoạn ngắn để tối ưu hóa việc mã hóa và phù hợp với giới hạn Context Window.         |
| **Context Window**      | Giới hạn số lượng tokens (đơn vị văn bản) tối đa mà LLM có thể tiếp nhận và xử lý trong một lần prompt.                     |
| **Grounding**           | Kỹ thuật "neo" câu trả lời của mô hình vào dữ liệu thực tế được cung cấp để đảm bảo tính xác thực.                          |

---

## I. Dẫn nhập

Sự bùng nổ của các Large Language Models (LLMs) như ChatGPT, Gemini, Claude hay Llama đã định hình lại lĩnh vực NLP. Tuy nhiên, dù sở hữu khả năng tổng quát và suy luận ấn tượng, các mô hình này vẫn đối mặt với những hạn chế cố hữu:

- Tri thức bị giới hạn tại thời điểm huấn luyện.
- Hiện tượng **hallucination** khi gặp các câu hỏi nằm ngoài vùng tri thức.
- Sự thiếu hụt kiến thức về dữ liệu riêng tư của doanh nghiệp.

Để giải quyết vấn đề này, kỹ thuật **Retrieval Augmented Generation (RAG)** đã ra đời. RAG cho phép LLMs tiếp cận nguồn dữ liệu bên ngoài mà không cần trải qua quá trình fine-tuning tốn kém hay phải huấn luyện lại.

Bài viết này bao gồm:

1. Phân tích sâu khái niệm, kiến trúc, pipeline cơ bản của RAG.
2. Giới thiệu framework LangChain, công cụ mạnh mẽ cho các ứng dụng sử dụng LLM.
3. Xây dựng hệ thống QA System trên tài liệu PDF học thuật.

> **Hình 1:** Minh hoạ LLM khi có (đường xanh lá) và không sử dụng RAG (đường màu đỏ).
>
> - Không dùng RAG → LLM trả lời sai: "10 module"
> - Dùng RAG với dữ liệu AIVN → LLM trả lời đúng: "12 module"

---

## II. Cơ sở lý thuyết về RAG

> **Nguồn gốc:** Khái niệm RAG lần đầu tiên được đề xuất chính thức trong bài báo khoa học _"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"_ bởi Patrick Lewis và các cộng sự tại Facebook AI Research (FAIR) vào năm 2020. Xem chi tiết tại: <https://arxiv.org/abs/2005.11401>

Trong công trình này, nhóm tác giả định nghĩa RAG là một **mô hình xác suất lai (hybrid probabilistic model)** kết hợp giữa hai dạng bộ nhớ:

- **Parametric Memory (Bộ nhớ tham số):** Là tri thức ẩn được lưu trữ trong trọng số của một mô hình sinh chuỗi (Pre-trained Seq2Seq Transformer). Cụ thể trong bài báo, tác giả sử dụng mô hình **BART** (Bidirectional and Auto-Regressive Transformers) làm Generator.
- **Non-Parametric Memory (Bộ nhớ phi tham số):** Là tri thức tường minh từ bên ngoài, cụ thể là một dense vector index chứa các đoạn văn bản Wikipedia. Thành phần này được truy cập thông qua một bộ **Neural Retriever** dựa trên kiến trúc Dense Passage Retriever.

Cơ chế hoạt động của RAG gốc cho phép Generator (BART) sử dụng đầu vào kết hợp với các tài liệu ẩn (latent documents) tìm được từ Retriever để sinh văn bản. Toàn bộ kiến trúc được **fine-tuning end-to-end**, cho phép cập nhật trọng số của cả Query Encoder và Generator.

---

### II.1. Sự chuyển dịch sang In-Context RAG

Mặc dù thuật ngữ RAG vẫn được giữ nguyên, tư duy triển khai kỹ thuật này đã thay đổi căn bản:

|                    | Original RAG (2020)                         | Modern RAG (Hiện nay)           |
| ------------------ | ------------------------------------------- | ------------------------------- |
| **Hướng tiếp cận** | Dựa trên fine-tuning                        | Dựa trên In-Context Learning    |
| **Cơ chế**         | Huấn luyện đồng thời retriever và generator | Quy trình "Retrieve and Prompt" |
| **Trọng số**       | Thay đổi trong quá trình huấn luyện         | Giữ nguyên trọng số LLM         |
| **Chi phí**        | Cao                                         | Thấp, linh hoạt                 |

Trong mô hình hiện đại, chúng ta thường **giữ nguyên trọng số của LLM** và chỉ tập trung tối ưu hóa việc truy xuất dữ liệu, sau đó đưa dữ liệu này vào đầu vào (Prompt) để mô hình xử lý.

---

### II.2. Kiến trúc RAG hiện đại

Một hệ thống RAG tiêu chuẩn hiện nay được mô hình hóa thành **3 giai đoạn chính**: Indexing, Retrieval và Generation.

```
Documents → Chunks → [Embedding Model] → Vector Store
                                              ↓
Query → [Embedding Model] → Query Vector → Similarity Search → Top-K Context → [LLM] → Answer
```

---

#### II.2.1. Phase 1: Indexing

Giai đoạn này tương đồng với quy trình **ETL (Extract-Transform-Load)** trong kỹ thuật dữ liệu. Mục tiêu là chuyển đổi dữ liệu thô thành định dạng có thể tìm kiếm được.

**1. Document Loading**

Bắt đầu với việc thu thập nguồn dữ liệu đầu vào.

- **Trích xuất nội dung:** Xử lý đa dạng các loại tệp tin, loại bỏ định dạng hiển thị phức tạp (font chữ, màu sắc, layout) và giữ lại văn bản thuần túy.
- **Thu thập siêu dữ liệu (metadata):** Trích xuất thông tin ngữ cảnh đi kèm tài liệu như chủ đề, số trang, ngày xuất bản, tác giả. Metadata hỗ trợ tính năng **Pre-filtering** — ví dụ: nếu người dùng hỏi về "Doanh thu năm 2024", hệ thống dùng metadata để lọc đúng các tài liệu trong năm 2024.

**2. Text Splitting (Chunking)**

Đây là bước chia nhỏ tài liệu dài thành các đoạn nhỏ hơn gọi là **"chunks"**.

_Tại sao cần Chunking?_

- **Giới hạn Context Window:** Các mô hình LLM và Embedding đều có giới hạn số token đầu vào.
- **Độ chính xác của tìm kiếm:** Vector của đoạn văn ngắn tập trung vào một ý cụ thể sẽ đại diện ngữ nghĩa tốt hơn vector của cả trang giấy nhiều chủ đề.

**Fixed-size Chunking:** Chia văn bản dựa trên số lượng ký tự/token cố định (ví dụ: 500 ký tự). Đơn giản nhưng dễ làm mất ngữ nghĩa nếu điểm cắt rơi vào giữa câu.

```
Ví dụ (chunk_size = 10 từ):
[Trí tuệ nhân tạo đang tạo ra những bước ngoặt] [lớn trong mọi lĩnh vực của đời sống hiện đại.]
```

**Recursive Chunking:** Phương pháp phổ biến nhất. Ưu tiên cắt theo: `\n\n` → `\n` → dấu chấm câu → khoảng trắng. Giúp giữ trọn vẹn cấu trúc câu và đoạn văn.

```
Ví dụ (Recursive bằng dấu chấm câu):
Chunk 1: "Trí tuệ nhân tạo đang tạo ra những bước ngoặt lớn trong mọi lĩnh vực của đời sống hiện đại."
Chunk 2: "Các mô hình ngôn ngữ lớn giúp chúng ta xử lý và phân tích khối lượng dữ liệu khổng lồ chỉ trong tích tắc."
Chunk 3: "Tuy nhiên, việc ứng dụng công nghệ này đòi hỏi sự hiểu biết sâu sắc để đảm bảo tính chính xác và an toàn."
```

**Chunk Overlap:** Thiết lập tham số `chunk_overlap` (thường 10–20% độ dài chunk) để đảm bảo ngữ nghĩa không bị mất tại điểm cắt.

> _Ví dụ:_ Chunk 1 kết thúc ở từ thứ 100, thì Chunk 2 sẽ bắt đầu từ từ thứ 80. Phần giao thoa đóng vai trò như "cầu nối" ngữ cảnh.

**3. Chiến lược Indexing nâng cao**

Ở RAG cơ bản, đoạn văn bản dùng để tìm kiếm và đoạn đưa cho LLM là cùng một. Tuy nhiên: _chunk nhỏ tốt cho tìm kiếm, chunk lớn tốt cho LLM_. Để giải quyết mâu thuẫn này:

- **Parent-Child Indexing (Small-to-Big):**
  - Chia văn bản thành **Parent chunks** lớn (~1000 tokens) để chứa ngữ cảnh đầy đủ.
  - Chia tiếp thành **Child chunks** nhỏ (~200 tokens).
  - **Cơ chế:** Index và tìm kiếm trên Child chunks (chính xác cao), nhưng trả về Parent chunk tương ứng cho LLM (ngữ cảnh đầy đủ).

- **Summary Indexing:** Sử dụng LLM để tóm tắt chunk gốc. Index phiên bản tóm tắt, nhưng trả về văn bản gốc chi tiết khi cần.

**4. Embedding**

- Sử dụng một mô hình Embedding để chuyển đổi các chunks thành **dense vector** trong không gian nhiều chiều.
- Các đoạn văn có nội dung ngữ nghĩa giống nhau sẽ có vector nằm gần nhau trong không gian này.

**5. Vector Store**

- Lưu trữ các vector cùng với ID và metadata vào cơ sở dữ liệu vector chuyên dụng để phục vụ truy xuất.

---

#### II.2.2. Phase 2: Retrieval

Đây là thành phần **quyết định sự thành bại** của hệ thống RAG. Nếu bước này trích xuất thông tin sai hoặc thiếu, LLM sẽ không có đủ dữ liệu để trả lời.

**1. Query Processing**

- Hệ thống nhận câu hỏi từ người dùng và đưa qua mô hình Embedding để tạo ra vector truy vấn _q_.
- Câu hỏi thực tế thường ngắn, thiếu ngữ cảnh. Các kỹ thuật tối ưu:
  - **Multi-Query:** Sử dụng LLM để sinh ra 3–5 biến thể khác nhau của câu hỏi gốc, tìm kiếm tất cả và gộp kết quả.

    ```
    Input (User): "Lỗi kết nối db"

    LLM Generated Queries:
    (a) "Cách khắc phục lỗi connection timeout khi kết nối database."
    (b) "Xử lý lỗi Access Denied cho user root trong MySQL/PostgreSQL."
    (c) "Hướng dẫn kiểm tra firewall chặn port 5432 hoặc 3306."

    → Hệ thống tìm kiếm cả 3 vấn đề (Timeout, Permission, Network).
    ```

  - **HyDE (Hypothetical Document Embeddings):** Yêu cầu LLM viết một câu trả lời giả định, sau đó dùng vector của câu trả lời giả định đó để tìm kiếm.

    ```
    Input (User): "Quy định mang laptop ra ngoài"

    Hypothetical Document (LLM giả định):
    "Theo chính sách bảo mật thông tin, nhân viên muốn mang thiết bị tài sản
    công ty ra khỏi văn phòng cần điền phiếu 'Đăng ký mang thiết bị' trên hệ
    thống và phải được Trưởng bộ phận phê duyệt..."

    → Vector của đoạn giả định khớp tốt hơn với "Sổ tay nhân viên" trong DB.
    ```

**2. Similarity Search**

- **Dense Retrieval:** Tính toán độ tương đồng giữa vector truy vấn _q_ và các vector tài liệu _d_.
  - **Metric:** Cosine Similarity hoặc Euclidean Distance.
  - **Algorithm:** Thuật toán **ANN (Approximate Nearest Neighbor)** như HNSW để tìm kiếm nhanh thay vì Brute-force _O(N)_.
- **Vấn đề:** Tìm kiếm vector tốt về ngữ nghĩa nhưng yếu về từ khóa chính xác (ví dụ: "iPhone 14" vs "iPhone 15").

**3. Hybrid Search**

Kết hợp hai luồng tìm kiếm song song:

| Phương pháp                         | Cơ chế                                       | Điểm mạnh                                         |
| ----------------------------------- | -------------------------------------------- | ------------------------------------------------- |
| **Dense Retrieval** (Vector Search) | Tìm kiếm dựa trên độ tương đồng ngữ nghĩa    | Bắt được từ đồng nghĩa, cách diễn đạt tương đương |
| **Sparse Retrieval** (BM25/TF-IDF)  | Đếm tần suất xuất hiện chính xác của từ khóa | Bắt chính xác tên riêng, thuật ngữ chuyên ngành   |

**Reciprocal Rank Fusion (RRF):** Thuật toán hậu xử lý để gộp và xếp hạng lại kết quả từ hai luồng.

> **Vấn đề:** Điểm số của Vector Search và BM25 có thang đo hoàn toàn khác nhau, việc cộng trực tiếp sẽ không hợp lí.
>
> **Giải pháp:** RRF bỏ qua điểm số gốc và chuẩn hóa dựa trên **thứ hạng**:
>
> $$\text{Score} = \sum_{i} \frac{1}{k + r_i}$$
>
> - _r_i_: Thứ hạng của tài liệu trong danh sách _i_ (hạng 1 → điểm lớn, hạng 100 → điểm nhỏ).
> - _k_: Hằng số (thường = 60), làm mượt điểm số, ngăn tài liệu Top-1 chiếm ưu thế tuyệt đối.
> - **Kết quả:** Tài liệu đứng hạng trung bình ở cả 2 danh sách thường có tổng điểm cao hơn tài liệu chỉ đứng nhất ở 1 danh sách.

**4. Re-ranking**

Sau khi có tập ứng viên (ví dụ: Top 50) từ bước trên, sử dụng **Cross-Encoder** để chấm điểm lại mức độ liên quan giữa câu hỏi và từng tài liệu.

|                 | Bi-Encoder                                        | Cross-Encoder                                                           |
| --------------- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| **Dùng ở bước** | Indexing / Retrieval                              | Re-ranking                                                              |
| **Cơ chế**      | Mã hóa câu hỏi và tài liệu thành 2 vector độc lập | Đưa cả câu hỏi và văn bản vào mô hình cùng lúc                          |
| **Ưu điểm**     | Cực nhanh                                         | Chính xác hơn, nhận biết sắc thái phủ định, quan hệ nguyên nhân-kết quả |
| **Nhược điểm**  | Mất mối quan hệ ngữ nghĩa phức tạp                | Chậm hơn nhiều                                                          |

**Chiến lược "Hình phễu":**

```
Retrieve Many (50 tài liệu — Bi-Encoder, nhanh)
    → Re-rank Few (5 tài liệu — Cross-Encoder, chính xác)
        → Đưa vào LLM
```

_Ví dụ minh họa Re-ranking:_

```
Query: "Tại sao tôi không nhận được thông báo qua email?"

Trước Re-ranking (Bi-Encoder):
  Rank 1: "Hướng dẫn cài đặt chữ ký email."        ← Sai ý định
  Rank 2: "Quy định về văn hóa gửi email công ty."  ← Sai
  Rank 3: "Xử lý lỗi email bị rơi vào thư mục Spam." ← Đúng nhưng bị xếp thấp

Sau Re-ranking (Cross-Encoder):
  Rank 1: "Xử lý lỗi email bị rơi vào thư mục Spam." ↑ Được đẩy lên đầu
  Rank 2: "Hướng dẫn cài đặt chữ ký email."
  Rank 3: "Quy định về văn hóa gửi email công ty."
```

---

#### II.2.3. Phase 3: Generation

Sau khi Retrieval trả về danh sách các tài liệu liên quan, LLM tổng hợp thông tin để trả lời người dùng.

**1. Context Preparation**

- **Context Stuffing:** Gộp toàn bộ văn bản Top-K tài liệu thành một đoạn dài và ghép vào Prompt.
  - _Vấn đề:_ Input dài → xử lý chậm, tốn token; quá nhiều thông tin không liên quan có thể làm LLM bị "loãng".

- **Context Reordering:** Dựa trên hiện tượng **"Lost in the Middle"** — LLM chú ý tốt nhất vào thông tin ở đầu và cuối prompt, hay bỏ quên thông tin ở giữa.

  > **Chiến thuật Reordering (dạng chữ U):**
  > Sắp xếp các tài liệu quan trọng nhất ở **hai đầu**, tài liệu ít quan trọng ở **giữa**:
  >
  > ```
  > Theo score:   Doc1 > Doc2 > Doc3 > Doc4 > Doc5
  > Sau reorder:  Doc1 | Doc3 | Doc5 | Doc4 | Doc2
  >               ^^^^ đầu             cuối ^^^^
  > ```

- **Context Compression:** Sử dụng LLM nhỏ hoặc thuật toán NLP để tóm tắt ý chính.

  ```
  Query: "Phí hủy dịch vụ là bao nhiêu?"

  Raw Context (300 tokens):
  "...Theo điều khoản 7.2 của hợp đồng dịch vụ được ký kết vào ngày...
  trong trường hợp khách hàng muốn chấm dứt hợp đồng trước thời hạn,
  cần thông báo bằng văn bản trước 30 ngày và chịu khoản phí phạt
  tương đương 02 tháng cước sử dụng..."

  Compressed Context (20 tokens):
  "Điều khoản 7.2: Phí hủy dịch vụ là 02 tháng cước sử dụng."
  ```

**2. Prompt Engineering**

- **Zero-shot Prompting:** Template cố định hướng dẫn mô hình trả lời trực tiếp dựa trên ngữ cảnh.

  ```
  System: You are an assistant for question-answering tasks. Use the
  following pieces of context to answer the question. If you don't know
  the answer, just say that you don't know.

  Context: {context}
  Question: {question}
  Answer:
  ```

- **Few-shot Learning:** Cung cấp thêm 1–2 ví dụ mẫu (Context - Question - Answer) để LLM học phong cách trả lời.

  ```
  Instruction: Trả lời ngắn gọn, chỉ đưa ra con số.

  Example 1:
  Context: "Doanh thu năm 2022 là 10 tỷ, năm 2023 tăng lên 12 tỷ."
  Question: Doanh thu 2023 bao nhiêu?
  Answer: 12 tỷ.

  Actual Task:
  Context: {real_context}
  Question: {user_question}
  Answer:
  ```

- **Chain-of-Thought (CoT):** Yêu cầu mô hình suy nghĩ từng bước (_"Let's think step by step"_) trước khi đưa ra kết luận.

  > Đối với câu hỏi phức tạp, nếu ép LLM trả lời ngay lập tức, nó dễ đoán mò. Với CoT, LLM tự sinh ra quy trình: _"Bước 1: Tìm giá gói A... Bước 2: Tìm giá gói B... Bước 3: So sánh..."_ → Kết luận chính xác hơn.
  >
  > Xem chi tiết: <https://arxiv.org/abs/2201.11903>

**3. Generation & Attribution**

- **Cơ bản:** LLM sinh câu trả lời dạng văn bản thông thường.
- **Citation (Trích dẫn nguồn):** Một lợi thế cạnh tranh của RAG là khả năng **minh bạch hóa nguồn tin**.
  - Trong prompt, yêu cầu LLM: _"Mọi thông tin đưa ra phải kèm theo ID của tài liệu nguồn"_.
  - Người dùng có thể click vào để mở tài liệu gốc đối chiếu.

  ```
  User Question: "Quy định làm thêm giờ tính thế nào?"

  RAG Response:
  "Theo quy định công ty, nhân viên làm thêm giờ vào ngày thường được hưởng
  150% lương cơ bản [Sổ tay nhân viên, Tr.12]. Đối với ngày lễ tết, mức hưởng
  là 300% lương [Luật lao động 2019, Điều 98]."
  ```

---

### II.3. Mở rộng: Phân tích vai trò của các thành phần

Phân tích theo câu hỏi: _"Hệ thống sẽ hoạt động như thế nào nếu loại bỏ một thành phần cụ thể?"_

**Kịch bản 1: Loại bỏ Embedding Model**

- **Vai trò:** Chuyển đổi văn bản sang vector để hỗ trợ truy vấn theo ngữ nghĩa.
- **Hệ quả:** Phải quay về Lexical Retrieval (BM25, TF-IDF, Exact Match). Giảm mạnh khả năng bắt từ đồng nghĩa và cách diễn đạt tương đương.
- _Ví dụ:_ Người dùng tìm "giá xe hơi" nhưng tài liệu chứa "chi phí mua ô tô" → bị bỏ qua vì không khớp từ khóa.

**Kịch bản 2: Loại bỏ Vector Store / ANN Index**

- **Vai trò:** Lưu trữ vector, quản lý metadata và hỗ trợ tìm kiếm nhanh ở quy mô lớn.
- **Hệ quả:** Phải dùng Brute-force _O(N)_ → độ trễ tăng mạnh khi dữ liệu lớn. Khó lọc metadata, khó cập nhật/xóa dữ liệu.

**Kịch bản 3: Loại bỏ LLM**

- **Vai trò:** Tổng hợp thông tin, suy luận và sinh câu trả lời dựa trên ngữ cảnh.
- **Hệ quả:** Output chỉ là danh sách đoạn văn (Top-K), người dùng phải tự đọc và rút kết luận. Hệ thống trở thành Semantic Search Engine, không phải chatbot.

**Tổng hợp:**

| Thành phần thiếu         | Hệ thống trở thành                 | Hạn chế cốt lõi                                          |
| ------------------------ | ---------------------------------- | -------------------------------------------------------- |
| Embedding Model          | Lexical Retrieval-driven RAG       | Giảm mạnh truy vấn theo ngữ nghĩa, khó bắt từ đồng nghĩa |
| Vector Store / ANN Index | Unscalable Prototype               | Độ trễ tăng mạnh ở dữ liệu lớn, khó vận hành             |
| LLM                      | Semantic Search / Retrieval System | Không thể tổng hợp, diễn giải, trả lời hội thoại         |

> _Bảng 1: Tổng hợp sự thay đổi của hệ thống RAG theo các kịch bản._
