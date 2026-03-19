"""PDF 解析器。"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from statistics import mean


def _load_fitz_module():
    """按需加载 PyMuPDF，避免在未安装时影响其他功能。"""

    try:
        return import_module("fitz")
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境缺少 PyMuPDF，先安装依赖后再解析 PDF。") from exc


@dataclass(frozen=True)
class PDFBlock:
    """PDF 中抽取出的结构化文本块。"""

    kind: str
    text: str
    page_number: int
    order: int
    font_size: float


@dataclass(frozen=True)
class PDFParseResult:
    """PDF 解析结果。"""

    source_name: str
    page_count: int
    blocks: tuple[PDFBlock, ...]

    @property
    def title_blocks(self) -> tuple[PDFBlock, ...]:
        """返回标题块。"""

        return tuple(block for block in self.blocks if block.kind == "title")

    @property
    def paragraph_blocks(self) -> tuple[PDFBlock, ...]:
        """返回正文块。"""

        return tuple(block for block in self.blocks if block.kind == "paragraph")

    @property
    def full_text(self) -> str:
        """拼接出完整文本。"""

        return "\n\n".join(block.text for block in self.blocks)


class PDFParser:
    """基于 PyMuPDF 的 PDF 解析器。"""

    def parse_file(self, pdf_path: str | Path) -> PDFParseResult:
        """解析本地 PDF 文件。"""

        file_path = Path(pdf_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF 路径不存在: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"PDF 路径不是文件: {file_path}")

        fitz = _load_fitz_module()
        try:
            with fitz.open(file_path) as document:
                return self._parse_document(
                    document=document,
                    source_name=file_path.name,
                )
        except FileNotFoundError:
            raise
        except Exception as exc:
            self._raise_pdf_error(exc)

    def parse_bytes(self, pdf_bytes: bytes, source_name: str = "uploaded.pdf") -> PDFParseResult:
        """解析上传得到的 PDF 二进制内容。"""

        if not pdf_bytes:
            raise ValueError("PDF 文件内容为空。")

        fitz = _load_fitz_module()
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                return self._parse_document(document=document, source_name=source_name)
        except Exception as exc:
            self._raise_pdf_error(exc)

    def _parse_document(self, document, source_name: str) -> PDFParseResult:
        """从已打开的文档中抽取结构化内容。"""

        if getattr(document, "needs_pass", False):
            raise ValueError("PDF 已加密，当前版本暂不支持带密码文档。")

        raw_blocks: list[dict[str, object]] = []
        for page_index, page in enumerate(document, start=1):
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                text = self._extract_block_text(block)
                if not text:
                    continue

                font_size = self._extract_font_size(block)
                raw_blocks.append(
                    {
                        "text": text,
                        "page_number": page_index,
                        "font_size": font_size,
                    }
                )

        classified_blocks = self._classify_blocks(raw_blocks)
        return PDFParseResult(
            source_name=source_name,
            page_count=len(document),
            blocks=tuple(classified_blocks),
        )

    def _classify_blocks(self, raw_blocks: list[dict[str, object]]) -> list[PDFBlock]:
        """用启发式规则把文本块区分为标题和正文。"""

        if not raw_blocks:
            return []

        font_sizes = [float(block["font_size"]) for block in raw_blocks]
        baseline_font_size = mean(font_sizes)
        max_font_size = max(font_sizes)

        classified_blocks: list[PDFBlock] = []
        for index, block in enumerate(raw_blocks):
            text = str(block["text"])
            font_size = float(block["font_size"])
            is_title = self._is_title_block(
                text=text,
                font_size=font_size,
                baseline_font_size=baseline_font_size,
                max_font_size=max_font_size,
            )
            classified_blocks.append(
                PDFBlock(
                    kind="title" if is_title else "paragraph",
                    text=text,
                    page_number=int(block["page_number"]),
                    order=index,
                    font_size=font_size,
                )
            )

        return classified_blocks

    def _is_title_block(
        self,
        text: str,
        font_size: float,
        baseline_font_size: float,
        max_font_size: float,
    ) -> bool:
        """用字号和文本形态判断当前块是否更像标题。"""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        normalized_text = " ".join(lines)
        text_length = len(normalized_text)
        ends_with_sentence_punctuation = normalized_text.endswith(
            ("。", "！", "？", ".", "!", "?", ";", "；", "：", ":")
        )
        starts_with_section_token = normalized_text.startswith(
            (
                "第",
                "Chapter",
                "CHAPTER",
                "Section",
                "SECTION",
                "Abstract",
                "ABSTRACT",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            )
        )
        has_short_shape = len(lines) <= 2 and text_length <= 80
        has_large_font = font_size >= baseline_font_size * 1.18 or font_size >= max_font_size * 0.92
        lacks_terminal_punctuation = not ends_with_sentence_punctuation

        if has_large_font and has_short_shape:
            return True
        if starts_with_section_token and has_short_shape and lacks_terminal_punctuation:
            return True
        return False

    def _extract_block_text(self, block: dict) -> str:
        """从 PyMuPDF 的 block 字典中提取纯文本。"""

        text_parts: list[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if line_text:
                text_parts.append(line_text)
        return "\n".join(text_parts).strip()

    def _extract_font_size(self, block: dict) -> float:
        """提取文本块平均字号。"""

        sizes: list[float] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size")
                if isinstance(size, (int, float)):
                    sizes.append(float(size))
        return mean(sizes) if sizes else 0.0

    def _raise_pdf_error(self, exc: Exception) -> None:
        """统一转换常见 PDF 解析异常。"""

        message = str(exc)
        if "password" in message.lower() or "encrypted" in message.lower():
            raise ValueError("PDF 已加密，当前版本暂不支持带密码文档。") from exc
        raise ValueError(f"PDF 解析失败: {message}") from exc
