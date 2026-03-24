"""独立论文导读页。"""

from __future__ import annotations

from html import escape

import streamlit as st

from labflow.ui.paper_preview import LandingPaperPreview
from labflow.ui.quick_guide import (
    LandingQuickGuide,
    build_landing_quick_guide,
    build_quick_guide_html,
    coerce_landing_quick_guide,
)


def render_quick_guide_page(
    *,
    preview: LandingPaperPreview | None,
    guide: LandingQuickGuide | None,
    source_name: str | None,
    has_repo_path: bool,
) -> None:
    """渲染独立的论文导读页。"""

    _render_quick_guide_page_header(
        source_name=source_name,
        has_repo_path=has_repo_path,
    )

    if preview is None:
        st.info("当前还没有可用的论文信息。先回到首页上传论文，再进入导读页。")
        return

    normalized_guide = (
        coerce_landing_quick_guide(guide, preview)
        if guide is not None
        else build_landing_quick_guide(preview, llm_client=None)
    )

    with st.container(border=True):
        st.markdown("### 论文概览")
        st.markdown(build_guide_page_overview_html(preview), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 导读结论")
        st.markdown(build_quick_guide_html(normalized_guide), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 摘要导读")
        if normalized_guide.abstract_digest:
            st.markdown(normalized_guide.abstract_digest)
        else:
            st.info("当前还没有生成摘要导读。")


def build_quick_guide_page_header_html(*, source_name: str | None) -> str:
    """导读页顶部紧凑头部。"""

    subtitle = f"当前论文：{source_name}" if source_name else "先在首页上传论文，再进入导读页。"
    return (
        '<div class="guide-page-shell">'
        '<div class="guide-page-kicker">LabFlow</div>'
        '<div class="guide-page-title">论文导读</div>'
        f'<div class="guide-page-body">{escape(subtitle)}</div>'
        "</div>"
    )


def _render_quick_guide_page_header(*, source_name: str | None, has_repo_path: bool) -> None:
    header_column, home_column, workspace_column = st.columns([6.4, 1.2, 1.4], gap="small")
    with header_column:
        st.markdown(
            build_quick_guide_page_header_html(source_name=source_name),
            unsafe_allow_html=True,
        )
        if not has_repo_path:
            st.caption("当前还没有填写代码路径，这一页先聚焦论文理解。")
    with home_column:
        st.write("")
        if st.button("首页", key="quick-guide-back-home"):
            st.session_state["current_route"] = "landing"
            st.rerun()
    with workspace_column:
        st.write("")
        if st.button(
            "工作区",
            disabled=not has_repo_path,
            key="quick-guide-to-workspace",
        ):
            st.session_state["current_route"] = "workspace"
            st.rerun()


def build_guide_page_overview_html(preview: LandingPaperPreview) -> str:
    """导读页的论文概览只保留中文用户需要的核心元信息。"""

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
        f'<div class="paper-preview-meta">{meta_html}</div>'
        f"{footer_html}"
        "</div>"
    )
