import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
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


def slugify(text: str) -> str:
    """
    Tạo slug ổn định cho ID (không dấu, lowercase, chỉ a-z0-9 và '-').
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


@dataclass
class CoursePdfStats:
    """Thống kê một môn (một thư mục con)."""

    course_folder: str
    pdf_count: int
    slide_pages: int  # tổng số trang PDF (coi như số slide)


@dataclass
class PdfCoursesLoadResult:
    documents: List = field(default_factory=list)
    courses: List[CoursePdfStats] = field(default_factory=list)
    """Thư mục môn có nhưng không có file .pdf (bị bỏ qua khi index)."""
    skipped_no_pdf: List[str] = field(default_factory=list)

    @property
    def num_courses(self) -> int:
        return len(self.courses)

    @property
    def total_slides(self) -> int:
        return sum(c.slide_pages for c in self.courses)

    @property
    def total_pdf_files(self) -> int:
        return sum(c.pdf_count for c in self.courses)


def format_pdf_courses_report(result: PdfCoursesLoadResult) -> str:
    n_folder = result.num_courses + len(result.skipped_no_pdf)
    lines = [
        "── Báo cáo PDF theo môn ──",
        f"Thư mục môn dưới data (tổng): {n_folder}",
        f"Số môn đã nạp được (có file .pdf): {result.num_courses}",
        f"Số file PDF đã nạp: {result.total_pdf_files}",
        f"Tổng số trang (slide): {result.total_slides}",
        "",
        "Chi tiết từng môn:",
    ]
    for c in sorted(result.courses, key=lambda x: x.course_folder):
        lines.append(
            f"  • {c.course_folder}: {c.pdf_count} file PDF, {c.slide_pages} trang"
        )
    if result.skipped_no_pdf:
        lines.append("")
        lines.append(
            "Không index (trong thư mục không có file .pdf — cần export PDF hoặc thêm loader .ppt):"
        )
        for name in sorted(result.skipped_no_pdf):
            lines.append(f"  ⊗ {name}")
    return "\n".join(lines)


class SimpleLoader:
    def _apply_location_metadata(
        self,
        doc,
        *,
        course_name: str | None,
        lecture_name: str | None,
        slide_path: str,
    ) -> None:
        meta = dict(getattr(doc, "metadata", {}) or {})

        page0 = meta.get("page", 0)
        try:
            page_number = int(page0) + 1  # 1-based
        except Exception:
            page_number = 1

        slide_file = Path(slide_path).name

        if course_name is not None:
            meta["course_name"] = course_name
            meta.setdefault("course", course_name)  # backward-compat
        if lecture_name is not None:
            meta["lecture_name"] = lecture_name
            if course_name is not None:
                meta["lecture_id"] = f"{slugify(course_name)}__{slugify(lecture_name)}"

        meta["slide_file"] = slide_file
        meta["slide_path"] = slide_path
        meta.setdefault("source", slide_path)  # backward-compat
        meta["page_number"] = page_number

        doc.metadata = meta

    def load_pdf(self, pdf_file: str):
        docs = PyPDFLoader(pdf_file).load()

        for doc in docs:
            doc.page_content = clean_vietnamese_text(doc.page_content)
            self._apply_location_metadata(
                doc,
                course_name=None,
                lecture_name=None,
                slide_path=str(pdf_file),
            )

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

    def load_pdf_courses(
        self,
        root_dir: str,
        *,
        recursive: bool = True,
    ) -> PdfCoursesLoadResult:
        """
        Nạp mọi PDF trong các thư mục con của root_dir (mỗi thư mục = một môn).
        Báo cáo và metadata dùng đúng tên thư mục trên đĩa (Path.name).
        Mỗi trang PDF được coi là một slide cho mục đích báo cáo.
        """
        from tqdm import tqdm

        root = Path(root_dir)
        if not root.is_dir():
            raise ValueError(f"Không phải thư mục hợp lệ: {root_dir}")

        result = PdfCoursesLoadResult()
        course_dirs = sorted(p for p in root.iterdir() if p.is_dir())

        for course_path in course_dirs:
            name = course_path.name

            if recursive:
                pdf_files = sorted(course_path.rglob("*.pdf"))
            else:
                pdf_files = sorted(course_path.glob("*.pdf"))

            if not pdf_files:
                result.skipped_no_pdf.append(name)
                continue

            course_docs: List = []
            short = name if len(name) <= 48 else name[:45] + "..."
            for pdf_file in tqdm(pdf_files, desc=f"PDF | {short}"):
                try:
                    docs = self.load_pdf(str(pdf_file))
                    for doc in docs:
                        rel = pdf_file.relative_to(course_path)
                        lecture_part = rel.parts[0] if len(rel.parts) > 1 else "default"
                        self._apply_location_metadata(
                            doc,
                            course_name=name,
                            lecture_name=lecture_part,
                            slide_path=str(pdf_file),
                        )
                    course_docs.extend(docs)
                except Exception:
                    pass

            if not course_docs and pdf_files:
                # Có file nhưng đọc hỏng — vẫn ghi nhận môn với 0 trang
                result.courses.append(
                    CoursePdfStats(
                        course_folder=name,
                        pdf_count=len(pdf_files),
                        slide_pages=0,
                    )
                )
                continue

            result.documents.extend(course_docs)
            result.courses.append(
                CoursePdfStats(
                    course_folder=name,
                    pdf_count=len(pdf_files),
                    slide_pages=len(course_docs),
                )
            )

        if not result.documents and not result.courses:
            raise ValueError(
                f"Không tìm thấy PDF trong thư mục môn nào dưới {root_dir}. "
                "Kiểm tra cấu trúc: root/pdf/<Tên môn>/*.pdf"
            )

        return result
