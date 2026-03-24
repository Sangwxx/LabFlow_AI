"""独立论文导读页。"""

from __future__ import annotations

from html import escape

import streamlit as st

from labflow.ui.paper_preview import LandingPaperPreview, build_paper_preview_html
from labflow.ui.quick_guide import LandingQuickGuide, build_quick_guide_html


def render_quick_guide_page(
    *,
    preview: LandingPaperPreview | None,
    guide: LandingQuickGuide | None,
    source_name: str | None,
    has_repo_path: bool,
) -> None:
    """渲染独立的论文导读页。"""

    st.markdown(
        build_quick_guide_page_header_html(source_name=source_name),
        unsafe_allow_html=True,
    )
    _render_quick_guide_page_actions(has_repo_path=has_repo_path)

    if preview is None:
        st.info("当前还没有可用的论文信息。先回到首页上传论文，再进入导读页。")
        return

    left_column, right_column = st.columns([1.08, 0.92], gap="medium")
    with left_column:
        with st.container(border=True):
            st.markdown("### 论文概览")
            st.markdown(build_paper_preview_html(preview), unsafe_allow_html=True)
    with right_column:
        with st.container(border=True):
            st.markdown("### 导读结论")
            if guide is not None:
                st.markdown(build_quick_guide_html(guide), unsafe_allow_html=True)
            else:
                st.info("当前还没有生成稳定的导读结论。")

    with st.container(border=True):
        st.markdown("### 建议怎么读")
        for item in build_quick_guide_reading_steps(preview=preview, guide=guide):
            st.markdown(f"- {item}")

    with st.container(border=True):
        st.markdown("### 摘要")
        st.markdown(preview.abstract or "当前摘要暂未识别，可直接进入工作区按段落阅读。")


def build_quick_guide_page_header_html(*, source_name: str | None) -> str:
    """导读页顶部头图。"""

    subtitle = f"当前论文：{source_name}" if source_name else "先在首页上传论文，再进入导读页。"
    return (
        '<div class="guide-page-shell">'
        '<div class="guide-page-kicker">LabFlow</div>'
        '<div class="guide-page-title">论文导读</div>'
        f'<div class="guide-page-body">{escape(subtitle)}</div>'
        "</div>"
    )


def build_quick_guide_reading_steps(
    *,
    preview: LandingPaperPreview,
    guide: LandingQuickGuide | None,
) -> tuple[str, ...]:
    """把导读页底部的阅读建议统一收成简短步骤。"""

    title = preview.title or "当前论文"
    base_steps = [
        f"先看《{title}》的标题、摘要和作者信息，确认任务背景与问题边界。",
        "再看方法部分，优先抓输入、核心模块和最终输出是怎么串起来的。",
        "最后再决定哪些段落需要进入工作区继续追到代码实现。",
    ]
    if guide is None:
        return tuple(base_steps)

    return (
        f"先确认：{guide.problem}",
        f"接着理解：{guide.method}",
        f"最后重点追踪：{guide.focus}",
    )


def _render_quick_guide_page_actions(*, has_repo_path: bool) -> None:
    left_action, right_action = st.columns([1, 1], gap="small")
    with left_action:
        if st.button("返回首页", use_container_width=True, key="quick-guide-back-home"):
            st.session_state["current_route"] = "landing"
            st.rerun()
    with right_action:
        if st.button(
            "进入工作区",
            use_container_width=True,
            disabled=not has_repo_path,
            key="quick-guide-to-workspace",
        ):
            st.session_state["current_route"] = "workspace"
            st.rerun()
        if not has_repo_path:
            st.caption("还没有填写代码路径时，导读页只负责看论文。")
