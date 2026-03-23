"""首页论文信息卡相关的纯函数。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

from labflow.clients.semantic_scholar_client import SemanticScholarPaper
from labflow.parsers.pdf_parser import PDFBlock, PDFParseResult

ARXIV_ID_PATTERN = re.compile(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
SECTION_HEADING_PATTERN = re.compile(r"^(?:abstract|references|\d+(?:\.\d+)*\b)", re.IGNORECASE)


@dataclass(frozen=True)
class LandingPaperPreview:
    """首页论文信息卡展示数据。"""

    title: str
    authors: tuple[str, ...]
    abstract: str
    source_label: str
    meta_items: tuple[str, ...]
    external_url: str | None = None


def build_landing_paper_preview(
    *,
    pdf_result: PDFParseResult,
    source_name: str,
    semantic_paper: SemanticScholarPaper | None = None,
) -> LandingPaperPreview | None:
    """优先基于本地 PDF 建卡，再用外部元数据补齐。"""

    local_title = extract_primary_paper_title(pdf_result) or Path(source_name).stem
    if not local_title:
        return None

    local_authors = extract_author_line(pdf_result)
    local_abstract = extract_abstract_text(pdf_result)
    arxiv_id = extract_arxiv_id(pdf_result)

    title = semantic_paper.title if semantic_paper and semantic_paper.title else local_title
    authors = semantic_paper.authors if semantic_paper and semantic_paper.authors else local_authors
    abstract = (
        semantic_paper.abstract if semantic_paper and semantic_paper.abstract else local_abstract
    )
    source_label = f"已识别论文 · {source_name}"
    meta_items = _build_meta_items(
        page_count=pdf_result.page_count,
        year=semantic_paper.year if semantic_paper else None,
        venue=semantic_paper.venue if semantic_paper else None,
        citation_count=semantic_paper.citation_count if semantic_paper else None,
        arxiv_id=arxiv_id,
    )
    return LandingPaperPreview(
        title=title,
        authors=authors[:6],
        abstract=abstract or "已识别论文标题，进入工作区后可继续按段落阅读与代码对齐。",
        source_label=source_label,
        meta_items=meta_items,
        external_url=semantic_paper.url if semantic_paper else None,
    )


def extract_primary_paper_title(pdf_result: PDFParseResult) -> str | None:
    """优先取首页最大字号、且不像章节标题的文本块。"""

    first_page_blocks = [block for block in pdf_result.blocks if block.page_number == 1]
    if not first_page_blocks:
        return None

    candidates = [block for block in first_page_blocks if _is_title_candidate(block)]
    if not candidates:
        return None

    selected = max(
        candidates,
        key=lambda block: (block.font_size, len(block.text)),
    )
    return " ".join(selected.text.split())


def extract_author_line(pdf_result: PDFParseResult) -> tuple[str, ...]:
    """尝试从首页标题下方抽取作者行。"""

    first_page_blocks = [block for block in pdf_result.blocks if block.page_number == 1]
    title = extract_primary_paper_title(pdf_result)
    if not title:
        return ()

    title_order = next(
        (block.order for block in first_page_blocks if " ".join(block.text.split()) == title),
        None,
    )
    if title_order is None:
        return ()

    for block in first_page_blocks:
        normalized_text = " ".join(block.text.split())
        if block.order <= title_order:
            continue
        if len(normalized_text) < 8 or len(normalized_text) > 180:
            continue
        if "@" in normalized_text or "http" in normalized_text.lower():
            continue
        if SECTION_HEADING_PATTERN.match(normalized_text):
            continue
        if normalized_text.count(",") + normalized_text.count(" and ") == 0:
            continue
        return tuple(
            part.strip(" ,") for part in re.split(r",| and ", normalized_text) if part.strip(" ,")
        )
    return ()


def extract_abstract_text(pdf_result: PDFParseResult) -> str:
    """从 Abstract 标题后抓取首段摘要。"""

    abstract_title = next(
        (
            block
            for block in pdf_result.title_blocks
            if block.page_number <= 2 and block.text.strip().lower().startswith("abstract")
        ),
        None,
    )
    if abstract_title is None:
        return ""

    following_blocks = [
        block
        for block in pdf_result.paragraph_blocks
        if block.page_number == abstract_title.page_number and block.order > abstract_title.order
    ]
    abstract_parts: list[str] = []
    for block in following_blocks:
        normalized_text = " ".join(block.text.split())
        if _looks_like_section_heading(normalized_text):
            break
        abstract_parts.append(normalized_text)
        if len(" ".join(abstract_parts)) >= 520:
            break

    return _truncate_text(" ".join(abstract_parts), 320)


def extract_arxiv_id(pdf_result: PDFParseResult) -> str | None:
    """从首页文本里提取 arXiv 编号。"""

    first_page_text = " ".join(block.text for block in pdf_result.blocks if block.page_number == 1)
    match = ARXIV_ID_PATTERN.search(first_page_text)
    if match is None:
        return None
    return match.group(1)


def build_paper_preview_html(preview: LandingPaperPreview) -> str:
    """把论文信息卡渲染成首页 HTML。"""

    meta_html = "".join(
        f'<span class="paper-preview-meta-chip">{escape(item)}</span>'
        for item in preview.meta_items
    )
    authors_text = " / ".join(preview.authors) if preview.authors else "作者信息待补全"
    footer_html = (
        f'<a class="paper-preview-link" href="{escape(preview.external_url)}" target="_blank">'
        "查看外部条目</a>"
        if preview.external_url
        else ""
    )
    return (
        '<div class="paper-preview-shell">'
        '<div class="paper-preview-head">'
        f'<div class="paper-preview-source">{escape(preview.source_label)}</div>'
        f'<div class="paper-preview-title">{escape(preview.title)}</div>'
        f'<div class="paper-preview-authors">{escape(authors_text)}</div>'
        "</div>"
        f'<div class="paper-preview-abstract">{escape(preview.abstract)}</div>'
        f'<div class="paper-preview-meta">{meta_html}</div>'
        f"{footer_html}"
        "</div>"
    )


def _is_title_candidate(block: PDFBlock) -> bool:
    normalized_text = " ".join(block.text.split())
    if len(normalized_text) < 12 or len(normalized_text) > 220:
        return False
    if "http" in normalized_text.lower():
        return False
    if ARXIV_ID_PATTERN.search(normalized_text):
        return False
    if SECTION_HEADING_PATTERN.match(normalized_text):
        return False
    return True


def _looks_like_section_heading(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) > 90:
        return False
    return bool(SECTION_HEADING_PATTERN.match(normalized))


def _truncate_text(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def _build_meta_items(
    *,
    page_count: int,
    year: int | None,
    venue: str | None,
    citation_count: int | None,
    arxiv_id: str | None,
) -> tuple[str, ...]:
    items = [f"{page_count} 页"]
    if year is not None:
        items.append(f"{year} 年")
    if venue:
        items.append(venue)
    if citation_count is not None:
        items.append(f"{citation_count} 次引用")
    if arxiv_id:
        items.append(f"arXiv:{arxiv_id}")
    return tuple(items)
