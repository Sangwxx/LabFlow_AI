"""首页论文信息卡相关测试。"""

from labflow.clients.semantic_scholar_client import SemanticScholarPaper
from labflow.parsers.pdf_parser import PDFBlock, PDFParseResult
from labflow.ui.paper_preview import (
    build_landing_paper_preview,
    build_paper_preview_html,
    extract_abstract_text,
    extract_author_line,
    extract_primary_paper_title,
)


def build_pdf_result() -> PDFParseResult:
    """构造一份足够覆盖首页论文信息卡的 PDF 解析结果。"""

    return PDFParseResult(
        source_name="Think.pdf",
        page_count=15,
        blocks=(
            PDFBlock(
                kind="paragraph",
                text=(
                    "Think Global, Act Local: Dual-scale Graph Transformer "
                    "for Vision-and-Language Navigation"
                ),
                page_number=1,
                order=0,
                font_size=14.35,
            ),
            PDFBlock(
                kind="paragraph",
                text=(
                    "Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, "
                    "Cordelia Schmid and Ivan Laptev"
                ),
                page_number=1,
                order=1,
                font_size=10.14,
            ),
            PDFBlock(
                kind="title",
                text="Abstract",
                page_number=1,
                order=2,
                font_size=11.96,
            ),
            PDFBlock(
                kind="paragraph",
                text=(
                    "We propose a dual-scale graph transformer for vision-and-language navigation. "
                    "The model jointly handles topological mapping and global action planning."
                ),
                page_number=1,
                order=3,
                font_size=9.8,
            ),
            PDFBlock(
                kind="title",
                text="1. Introduction",
                page_number=1,
                order=4,
                font_size=11.96,
            ),
            PDFBlock(
                kind="paragraph",
                text="arXiv:2202.11742v1 [cs.CV] 23 Feb 2022",
                page_number=1,
                order=5,
                font_size=20.0,
            ),
        ),
    )


def test_extract_primary_paper_title_prefers_first_page_main_title() -> None:
    """首页标题提取应优先命中首页真正的论文标题，而不是章节标题。"""

    title = extract_primary_paper_title(build_pdf_result())

    assert (
        title == "Think Global, Act Local: Dual-scale Graph Transformer "
        "for Vision-and-Language Navigation"
    )


def test_extract_author_line_reads_title_following_block() -> None:
    """作者行应从标题下方的第一条有效文本里提取。"""

    authors = extract_author_line(build_pdf_result())

    assert authors[:2] == ("Shizhe Chen", "Pierre-Louis Guhur")
    assert authors[-1] == "Ivan Laptev"


def test_extract_abstract_text_stops_before_next_section() -> None:
    """摘要提取应在下一个章节标题前停止。"""

    abstract = extract_abstract_text(build_pdf_result())

    assert "dual-scale graph transformer" in abstract
    assert "Introduction" not in abstract


def test_build_landing_paper_preview_prefers_remote_metadata_when_available() -> None:
    """外部元数据可用时，首页卡片应优先展示更完整的信息。"""

    preview = build_landing_paper_preview(
        pdf_result=build_pdf_result(),
        source_name="Think.pdf",
        semantic_paper=SemanticScholarPaper(
            title=(
                "Think Global, Act Local: Dual-scale Graph Transformer "
                "for Vision-and-Language Navigation"
            ),
            authors=("Author A", "Author B"),
            abstract="Semantic Scholar abstract",
            year=2022,
            citation_count=123,
            venue="CVPR",
            url="https://example.com/paper",
        ),
    )

    assert preview is not None
    assert preview.authors == ("Author A", "Author B")
    assert preview.abstract == "Semantic Scholar abstract"
    assert "2022 年" in preview.meta_items
    assert "123 次引用" in preview.meta_items


def test_build_paper_preview_html_contains_title_and_meta() -> None:
    """论文信息卡 HTML 应保留标题、作者和元数据标签。"""

    preview = build_landing_paper_preview(
        pdf_result=build_pdf_result(),
        source_name="Think.pdf",
    )

    assert preview is not None
    html = build_paper_preview_html(preview)

    assert "已识别论文" in html
    assert "Think Global, Act Local" in html
    assert "Shizhe Chen" in html
    assert "15 页" in html
    assert "arXiv:2202.11742v1" in html
