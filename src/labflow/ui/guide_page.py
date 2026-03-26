"""独立论文导读页渲染。"""

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
    has_llm_credentials: bool,
) -> None:
    """渲染独立的论文导读页。"""

    _render_quick_guide_page_header(
        source_name=source_name,
        has_repo_path=has_repo_path,
        has_llm_credentials=has_llm_credentials,
    )

    if preview is None:
        st.info("当前还没有可导读的论文。请先回到首页上传 PDF，再进入导读页。")
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
        st.markdown("### 导读报告")
        st.markdown(build_quick_guide_html(normalized_guide), unsafe_allow_html=True)


def build_quick_guide_page_header_html(*, source_name: str | None) -> str:
    """构建导读页头部。"""

    subtitle = f"当前论文：{source_name}" if source_name else "当前还没有加载论文"
    return (
        '<div class="guide-page-shell">'
        '<div class="guide-page-kicker">LabFlow</div>'
        '<div class="guide-page-title">论文导读</div>'
        f'<div class="guide-page-body">{escape(subtitle)}</div>'
        "</div>"
    )


def build_guide_page_status_text(*, has_repo_path: bool, has_llm_credentials: bool) -> str:
    """根据当前运行时状态生成顶部提示。"""

    if has_repo_path and has_llm_credentials:
        return "代码目录已连接，可直接进入工作区继续定位源码实现。"
    if has_repo_path:
        return "代码目录已连接；当前未配置模型 API Key，导读会优先使用本地兜底。"
    if has_llm_credentials:
        return "当前还没有填写代码路径，这一页先聚焦论文理解。"
    return "当前还没有填写代码路径，也未配置模型 API Key；导读会优先使用本地兜底。"


def _render_quick_guide_page_header(
    *,
    source_name: str | None,
    has_repo_path: bool,
    has_llm_credentials: bool,
) -> None:
    title_column, action_column = st.columns([6.4, 3.6], gap="medium")
    with title_column:
        st.markdown(
            build_quick_guide_page_header_html(source_name=source_name),
            unsafe_allow_html=True,
        )
    with action_column:
        status_text = build_guide_page_status_text(
            has_repo_path=has_repo_path,
            has_llm_credentials=has_llm_credentials,
        )
        st.markdown(
            (f'<div class="guide-page-toolbar-note">{escape(status_text)}</div>'),
            unsafe_allow_html=True,
        )
        home_column, workspace_column = st.columns(2, gap="small")
        with home_column:
            if st.button("返回首页", use_container_width=True, key="quick-guide-back-home"):
                st.session_state["current_route"] = "landing"
                st.rerun()
        with workspace_column:
            if st.button(
                "进入工作区",
                use_container_width=True,
                disabled=not has_repo_path,
                key="quick-guide-to-workspace",
            ):
                st.session_state["current_route"] = "workspace"
                st.rerun()


def build_guide_page_overview_html(preview: LandingPaperPreview) -> str:
    """构建导读页论文概览区，不直接暴露原始英文摘要。"""

    meta_html = "".join(
        f'<span class="paper-preview-meta-chip">{escape(item)}</span>'
        for item in preview.meta_items
    )
    authors_text = " / ".join(preview.authors) if preview.authors else "暂无作者信息"
    footer_html = (
        f'<a class="paper-preview-link" href="{escape(preview.external_url)}" target="_blank">'
        "查看外部条目</a>"
        if preview.external_url
        else ""
    )
    return (
        '<div class="paper-preview-shell guide-page-overview-shell">'
        '<div class="paper-preview-head">'
        f'<div class="paper-preview-source">{escape(preview.source_label)}</div>'
        f'<div class="paper-preview-title">{escape(preview.title)}</div>'
        f'<div class="paper-preview-authors">{escape(authors_text)}</div>'
        "</div>"
        f'<div class="paper-preview-meta">{meta_html}</div>'
        f"{footer_html}"
        "</div>"
    )
