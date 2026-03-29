import re
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

        # Chuẩn hoá nhẹ để in ra gọn, vẫn giữ xuống dòng để dễ đọc + giữ citations.
        answer = re.sub(r"[ \t]+\n", "\n", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        answer = answer.strip()

        return answer


def _u_shape_reorder(docs):
    # Doc1 Doc2 Doc3 Doc4 Doc5 -> Doc1 Doc3 Doc5 Doc4 Doc2
    odds = docs[0::2]
    evens = docs[1::2][::-1]
    return odds + evens


def _doc_citation_label(doc) -> str:
    meta = dict(getattr(doc, "metadata", {}) or {})
    course = meta.get("course_name") or meta.get("course") or ""
    lecture = meta.get("lecture_name") or ""
    slide = meta.get("slide_file") or ""
    page = meta.get("page_number")
    if page is None:
        try:
            page = int(meta.get("page", 0)) + 1
        except Exception:
            page = 1
    return f"{course}|{lecture}|{slide}|p{int(page)}"


def format_docs(docs):
    docs = _u_shape_reorder(list(docs))
    blocks = []
    seen = set()
    for doc in docs:
        content = (getattr(doc, "page_content", "") or "").strip()
        if not content:
            continue
        # tránh nhồi trùng nội dung y hệt
        if content in seen:
            continue
        seen.add(content)
        label = _doc_citation_label(doc)
        blocks.append(f"[{label}]\n{content}")
    return "\n\n".join(blocks)


def format_docs_with_source_ids(docs):
    """
    Trả về (context_text, id_to_label).

    Ý tưởng: ép model chỉ được cite [S1], [S2]... (dễ kiểm soát),
    rồi map lại sang nhãn thật [course|lecture|slide|pX] để nguồn luôn chính xác.
    """
    docs = _u_shape_reorder(list(docs))
    blocks = []
    seen = set()
    id_to_label: dict[str, str] = {}
    i = 0

    for doc in docs:
        content = (getattr(doc, "page_content", "") or "").strip()
        if not content:
            continue
        if content in seen:
            continue
        seen.add(content)

        i += 1
        sid = f"S{i}"
        label = _doc_citation_label(doc)
        id_to_label[sid] = label
        blocks.append(f"[{sid}] [{label}]\n{content}")

    return ("\n\n".join(blocks), id_to_label)


class OfflineRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        self.prompt = PromptTemplate.from_template("""
Bạn là trợ lý AI.

Chỉ sử dụng thông tin trong context để trả lời. Nếu context không có thông tin, hãy trả lời đúng nguyên văn: "Không có thông tin".

Yêu cầu:
- Trả lời rõ ràng, không lặp ý.
- Trả lời theo gạch đầu dòng (3–7 ý) nếu phù hợp.
- Không tự giới thiệu, không viết kiểu “trang chủ/diễn đàn/tài liệu đang tìm kiếm…”. Đi thẳng vào câu trả lời.
- Mỗi ý quan trọng phải kèm trích dẫn nguồn.
- Bạn CHỈ được phép trích dẫn theo dạng [S1], [S2], ... (copy y nguyên các mã S có trong context).
- Tuyệt đối KHÔNG tự chế trích dẫn kiểu [course|...], [lecture|...], hoặc bất kỳ nhãn nào khác ngoài [S<number>].

Context:
{context}

Câu hỏi:
{question}

[TRẢ LỜI]:
""")

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | FocusedAnswerParser()
        )

    def ask(self, query: str):
        # Làm thủ công để đảm bảo citations khớp nguồn (S1..Sn -> label thật).
        docs = list(self.retriever.invoke(query) or [])
        context, id_to_label = format_docs_with_source_ids(docs)

        prompt_text = self.prompt.format(context=context, question=query)
        raw = self.llm.invoke(prompt_text)
        answer = FocusedAnswerParser().parse(str(raw))

        if answer.strip() == "Không có thông tin":
            return "Không có thông tin"

        # Lấy các mã nguồn model đã dùng (chỉ chấp nhận S<number>).
        used_ids = re.findall(r"\[S(\d+)\]", answer)
        used_sids = []
        seen = set()
        for n in used_ids:
            sid = f"S{int(n)}"
            if sid in id_to_label and sid not in seen:
                seen.add(sid)
                used_sids.append(sid)

        # Thay [Sx] -> [course|lecture|slide|pX] để người dùng thấy nguồn thật.
        def _replace_sid(m: re.Match) -> str:
            n = int(m.group(1))
            sid = f"S{n}"
            label = id_to_label.get(sid)
            return f"[{label}]" if label else ""

        answer = re.sub(r"\[S(\d+)\]", _replace_sid, answer)

        # Gỡ các bracket lạ còn sót (để tránh model tự chế nhãn khác).
        answer = re.sub(r"\[(?![^\[\]\|]+\|[^\[\]\|]*\|[^\[\]\|]*\|p\d+\])[^\[\]]+\]", "", answer)
        answer = FocusedAnswerParser().parse(answer)

        # Append danh sách nguồn đã dùng (nếu có).
        if used_sids:
            src_lines = [f"- [{id_to_label[sid]}]" for sid in used_sids if sid in id_to_label]
            if src_lines and "Nguồn:" not in answer:
                answer = answer.rstrip() + "\n\nNguồn:\n" + "\n".join(src_lines)

        return answer.strip()