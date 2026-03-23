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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class OfflineRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        self.prompt = PromptTemplate.from_template("""
Bạn là trợ lý AI.

Chỉ sử dụng thông tin trong context để trả lời.

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