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

        answer = re.sub(r"^\s*[\-\*]\s*", "", answer, flags=re.MULTILINE)
        answer = re.sub(r"\n+", " ", answer)

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


class OfflineRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        self.prompt = PromptTemplate.from_template("""
Bạn là trợ lý AI.

Chỉ sử dụng thông tin trong context để trả lời. Nếu context không có thông tin, hãy trả lời đúng nguyên văn: "Không có thông tin".

Yêu cầu:
- Trả lời ngắn gọn.
- Mỗi ý quan trọng phải kèm trích dẫn nguồn theo định dạng [course|lecture|slide|pX] lấy từ nhãn trong context.

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
        return self.chain.invoke(query)