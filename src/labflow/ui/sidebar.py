"""侧边栏输入组件。"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class SidebarState:
    """侧边栏输入状态。"""

    uploaded_pdf_name: str | None
    uploaded_pdf_bytes: bytes | None
    git_repo_path: str


def render_sidebar() -> SidebarState:
    """渲染阶段 1 所需的侧边栏输入组件。"""

    with st.sidebar:
        st.markdown("## 感知层输入")
        uploaded_pdf = st.file_uploader(
            "上传论文 PDF",
            type=["pdf"],
            help="首版先支持文本型 PDF，后续再补扫描版 OCR。",
        )
        git_repo_path = st.text_input(
            "本地 Git 仓库路径",
            placeholder=r"D:\projects\demo-repo",
            help="支持输入仓库根目录，或仓库中的任意子目录。",
        ).strip()

        pdf_bytes = uploaded_pdf.getvalue() if uploaded_pdf is not None else None
        pdf_name = uploaded_pdf.name if uploaded_pdf is not None else None
        return SidebarState(
            uploaded_pdf_name=pdf_name,
            uploaded_pdf_bytes=pdf_bytes,
            git_repo_path=git_repo_path,
        )
