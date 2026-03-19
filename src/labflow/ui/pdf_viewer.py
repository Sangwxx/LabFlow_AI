"""PDF 预览组件。"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import declare_component

from labflow.parsers.pdf_parser import PDFBlock

COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "pdf_hotspot"
PDF_HOTSPOT_COMPONENT = declare_component("pdf_hotspot", path=str(COMPONENT_DIR))


@dataclass(frozen=True)
class PDFHotspot:
    """页图上的可点击段落热区。"""

    block_order: int
    page_number: int
    top_percent: float
    left_percent: float
    width_percent: float
    height_percent: float
    label: str


def render_pdf_viewer(
    pdf_bytes: bytes,
    *,
    blocks: tuple[PDFBlock, ...] = (),
    height: int = 920,
    page_number: int | None = None,
    selected_block_order: int | None = None,
    key: str = "pdf_hotspot_viewer",
) -> int | None:
    """把 PDF 渲染成页图，并通过本地组件把热区点击安全回传给 Python。"""

    rendered_pages = _render_pdf_pages(pdf_bytes)
    if not rendered_pages:
        st.warning("当前 PDF 暂时无法渲染预览。")
        return None

    target_page = page_number if page_number and 1 <= page_number <= len(rendered_pages) else 1
    hotspots = _build_hotspots(blocks)
    selected_value = PDF_HOTSPOT_COMPONENT(
        pages=rendered_pages,
        hotspots=[hotspot.__dict__ for hotspot in hotspots],
        height=height,
        target_page=target_page,
        selected_block_order=selected_block_order,
        default=selected_block_order,
        key=key,
    )
    if selected_value is None:
        return None
    try:
        return int(selected_value)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def _render_pdf_pages(pdf_bytes: bytes) -> tuple[str, ...]:
    """把 PDF 每一页渲染成 PNG，优先保证工作区内预览稳定。"""

    fitz = _load_fitz_module()
    rendered_pages: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
            rendered_pages.append(base64.b64encode(pixmap.tobytes("png")).decode("utf-8"))
    return tuple(rendered_pages)


def _build_hotspots(blocks: tuple[PDFBlock, ...]) -> tuple[PDFHotspot, ...]:
    """把正文块坐标转换成页图热区。"""

    hotspots: list[PDFHotspot] = []
    for block in blocks:
        if block.kind != "paragraph":
            continue

        x0, y0, x1, y1 = block.bbox
        width = max(block.page_width, 1.0)
        height = max(block.page_height, 1.0)
        hotspots.append(
            PDFHotspot(
                block_order=block.order,
                page_number=block.page_number,
                top_percent=max(0.0, y0 / height * 100),
                left_percent=max(0.0, x0 / width * 100),
                width_percent=max(0.6, (x1 - x0) / width * 100),
                height_percent=max(1.4, (y1 - y0) / height * 100),
                label=block.text[:120].replace('"', "'"),
            )
        )
    return tuple(hotspots)


def _load_fitz_module():
    """按需加载 PyMuPDF，保持和解析器一致的依赖约束。"""

    try:
        return import_module("fitz")
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境缺少 PyMuPDF，先安装依赖后再预览 PDF。") from exc
