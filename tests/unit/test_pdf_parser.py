"""PDF 解析器测试。"""

from io import BytesIO
from types import SimpleNamespace

import pytest

from labflow.parsers.pdf_parser import PDFParser


class FakePage:
    """伪造的 PDF 页面。"""

    def __init__(self, blocks: list[dict], *, width: float = 595.0, height: float = 842.0) -> None:
        self._blocks = blocks
        self.rect = SimpleNamespace(width=width, height=height)

    def get_text(self, mode: str) -> dict:
        """返回伪造的 PyMuPDF `dict` 结构。"""

        assert mode == "dict"
        return {"blocks": self._blocks}


class FakeDocument:
    """伪造的 PDF 文档。"""

    def __init__(self, pages: list[FakePage], needs_pass: bool = False) -> None:
        self._pages = pages
        self.needs_pass = needs_pass

    def __iter__(self):
        return iter(self._pages)

    def __len__(self) -> int:
        return len(self._pages)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def build_text_block(
    text: str,
    font_size: float,
    *,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 200.0, 60.0),
) -> dict:
    """构造一段简化的文本块。"""

    return {
        "bbox": bbox,
        "lines": [
            {
                "spans": [
                    {
                        "text": text,
                        "size": font_size,
                    }
                ]
            }
        ],
    }


def build_multiline_text_block(
    lines: list[str],
    font_size: float,
    *,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 200.0, 60.0),
) -> dict:
    """构造一个带多行文本的 PDF block。"""

    return {
        "bbox": bbox,
        "lines": [
            {
                "spans": [
                    {
                        "text": line,
                        "size": font_size,
                    }
                ]
            }
            for line in lines
        ],
    }


def test_pdf_parser_classifies_title_and_paragraph(monkeypatch) -> None:
    """我会用字号和文本形态把标题与正文区分开。"""

    fake_document = FakeDocument(
        pages=[
            FakePage(
                [
                    build_text_block("1 Introduction", 18),
                    build_text_block(
                        "This paper presents a practical workflow "
                        "for aligning paper sections with code.",
                        11,
                    ),
                ]
            )
        ]
    )
    fake_fitz = SimpleNamespace(open=lambda **_: fake_document)
    monkeypatch.setattr("labflow.parsers.pdf_parser._load_fitz_module", lambda: fake_fitz)

    parser = PDFParser()
    result = parser.parse_bytes(b"demo-pdf", source_name="demo.pdf")

    assert result.page_count == 1
    assert len(result.title_blocks) == 1
    assert result.title_blocks[0].text == "1 Introduction"
    assert result.title_blocks[0].bbox == (10.0, 20.0, 200.0, 60.0)
    assert result.title_blocks[0].page_width == 595.0
    assert len(result.paragraph_blocks) == 1


def test_pdf_parser_rejects_encrypted_document(monkeypatch) -> None:
    """遇到加密 PDF 时我要给出明确错误。"""

    fake_document = FakeDocument(pages=[FakePage([])], needs_pass=True)
    fake_fitz = SimpleNamespace(open=lambda **_: fake_document)
    monkeypatch.setattr("labflow.parsers.pdf_parser._load_fitz_module", lambda: fake_fitz)

    parser = PDFParser()

    with pytest.raises(ValueError, match="已加密"):
        parser.parse_bytes(b"demo-pdf", source_name="locked.pdf")


def test_pdf_parser_supports_bytesio_stream(monkeypatch) -> None:
    """上传组件给我的是内存流时，我也要能正常读取。"""

    fake_document = FakeDocument(
        pages=[
            FakePage(
                [
                    build_text_block("2 Method", 17),
                    build_text_block("The method uses a streamed PDF input.", 11),
                ]
            )
        ]
    )
    fake_fitz = SimpleNamespace(open=lambda **_: fake_document)
    monkeypatch.setattr("labflow.parsers.pdf_parser._load_fitz_module", lambda: fake_fitz)

    parser = PDFParser()
    result = parser.parse_stream(BytesIO(b"demo-pdf"), source_name="stream.pdf")

    assert result.source_name == "stream.pdf"
    assert result.title_blocks[0].text == "2 Method"


def test_pdf_parser_normalizes_linebreak_and_hyphenation(monkeypatch) -> None:
    """跨行断词和换行应尽量还原成自然段文本。"""

    fake_document = FakeDocument(
        pages=[
            FakePage(
                [
                    build_text_block("Abstract", 17),
                    build_multiline_text_block(
                        [
                            "Following language instructions to navigate in unseen",
                            "environments is a challenging problem for autonomous em-",
                            "bodied agents.",
                        ],
                        11,
                    ),
                ]
            )
        ]
    )
    fake_fitz = SimpleNamespace(open=lambda **_: fake_document)
    monkeypatch.setattr("labflow.parsers.pdf_parser._load_fitz_module", lambda: fake_fitz)

    result = PDFParser().parse_bytes(b"demo-pdf", source_name="normalized.pdf")

    assert result.paragraph_blocks[0].text == (
        "Following language instructions to navigate in unseen "
        "environments is a challenging problem for autonomous embodied agents."
    )


def test_pdf_parser_requires_existing_file() -> None:
    """文件路径不存在时应直接失败。"""

    parser = PDFParser()

    with pytest.raises(FileNotFoundError):
        parser.parse_file(".tmp/tests/not-found.pdf")
