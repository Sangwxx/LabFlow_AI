"""首页入口视图。"""

from __future__ import annotations

from html import escape

import streamlit as st


def render_landing() -> None:
    has_pdf = bool(st.session_state.get("landing_pdf_bytes"))
    has_repo_path = bool(st.session_state.get("landing_git_repo_path", "").strip())
    _, content_column, _ = st.columns([0.7, 4.6, 0.7])
    with content_column:
        st.markdown(
            build_landing_hero_html(has_pdf=has_pdf, has_repo_path=has_repo_path),
            unsafe_allow_html=True,
        )

        pdf_column, git_column = st.columns(2, gap="medium")
        uploaded_pdf = _render_pdf_input_card(pdf_column, has_pdf=has_pdf)
        git_repo_path = _render_repo_input_card(git_column, has_repo_path=has_repo_path)

        if uploaded_pdf is not None:
            st.session_state["landing_pdf_bytes"] = uploaded_pdf.getvalue()
            st.session_state["landing_pdf_name"] = uploaded_pdf.name
        st.session_state["landing_git_repo_path"] = git_repo_path

        _render_landing_action(has_pdf=has_pdf, has_repo_path=has_repo_path)


def _render_pdf_input_card(column, *, has_pdf: bool):
    with column:
        with st.container(border=True):
            st.markdown(
                build_landing_entry_header_html(
                    step_label="步骤 1",
                    title="论文 PDF",
                    description="上传后即可开始按段落阅读。",
                    status_text="已选择" if has_pdf else "未选择",
                    status_tone="ready" if has_pdf else "pending",
                ),
                unsafe_allow_html=True,
            )
            uploaded_pdf = st.file_uploader(
                "选择 PDF",
                type=["pdf"],
                key="landing_pdf_uploader",
                label_visibility="collapsed",
            )
            current_pdf_name = (
                uploaded_pdf.name
                if uploaded_pdf is not None
                else st.session_state.get("landing_pdf_name")
            )
            if current_pdf_name:
                st.caption(f"当前文件：`{current_pdf_name}`")
    return uploaded_pdf


def _render_repo_input_card(column, *, has_repo_path: bool) -> str:
    with column:
        with st.container(border=True):
            st.markdown(
                build_landing_entry_header_html(
                    step_label="步骤 2",
                    title="代码目录",
                    description="支持 Git 仓库或普通 Python 目录。",
                    status_text="已填写" if has_repo_path else "未填写",
                    status_tone="ready" if has_repo_path else "pending",
                ),
                unsafe_allow_html=True,
            )
            git_repo_path = st.text_input(
                "代码路径",
                value=st.session_state.get("landing_git_repo_path", ""),
                placeholder=r"E:\project\your-repo",
                key="landing_git_repo_path_input",
                label_visibility="collapsed",
            ).strip()
            st.caption("例如：`E:\\VLN-DUET-main`")
    return git_repo_path


def _render_landing_action(*, has_pdf: bool, has_repo_path: bool) -> None:
    _, action_center, _ = st.columns([1.25, 0.9, 1.25])
    with action_center:
        st.markdown(
            (
                '<div class="landing-action-hint">'
                f"{escape(build_landing_readiness_text(has_pdf=has_pdf, has_repo_path=has_repo_path))}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if st.button("进入工作区", type="primary", use_container_width=True):
            if not st.session_state.get("landing_pdf_bytes"):
                st.warning("先上传论文 PDF，再进入工作区。")
                return
            if not st.session_state.get("landing_git_repo_path"):
                st.warning("先填写本地代码路径，再进入工作区。")
                return
            st.session_state["current_route"] = "workspace"
            st.session_state["selected_section_index"] = None
            st.session_state["pdf_hotspot_viewer"] = None
            st.rerun()


def build_landing_hero_html(*, has_pdf: bool, has_repo_path: bool) -> str:
    state_text = build_landing_readiness_text(has_pdf=has_pdf, has_repo_path=has_repo_path)
    return (
        '<div class="landing-shell">'
        '<div class="landing-kicker">LabFlow</div>'
        '<div class="landing-title">论文与代码，对齐阅读</div>'
        '<div class="landing-body">准备好 PDF 和代码目录后，直接进入工作区。</div>'
        f'<div class="landing-status-line">{escape(state_text)}</div>'
        "</div>"
    )


def build_landing_entry_header_html(
    *,
    step_label: str,
    title: str,
    description: str,
    status_text: str,
    status_tone: str,
) -> str:
    status_class = (
        "entry-card-state entry-card-state-ready"
        if status_tone == "ready"
        else "entry-card-state"
    )
    return (
        '<div class="entry-card-head">'
        '<div class="entry-card-meta-row">'
        f'<div class="entry-card-step">{escape(step_label)}</div>'
        f'<div class="{status_class}">{escape(status_text)}</div>'
        "</div>"
        f'<div class="entry-card-title">{escape(title)}</div>'
        f'<div class="entry-card-desc">{escape(description)}</div>'
        "</div>"
    )


def build_landing_readiness_text(*, has_pdf: bool, has_repo_path: bool) -> str:
    if has_pdf and has_repo_path:
        return "已准备好，可以开始阅读。"
    if has_pdf:
        return "还差代码目录。"
    if has_repo_path:
        return "还差论文 PDF。"
    return "先完成这两项输入。"
