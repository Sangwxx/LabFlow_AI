"""侧边栏输入组件。"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from labflow.config.settings import Settings

DEFAULT_MODEL_OPTIONS = (
    "moonshotai/kimi-k2.5",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4",
    "qwen3-coder-480b-a35b-instruct",
)


@dataclass(frozen=True)
class SidebarState:
    """侧边栏输入状态。"""

    uploaded_pdf_name: str | None
    uploaded_pdf_bytes: bytes | None
    git_repo_path: str
    api_key: str | None
    base_url: str
    model_name: str


def render_sidebar(settings: Settings) -> SidebarState:
    """渲染联调模式下的侧边栏配置。"""

    with st.sidebar:
        st.markdown("## 数据输入")
        uploaded_pdf = st.file_uploader(
            "上传论文 PDF",
            type=["pdf"],
            help="支持直接读取上传文件的内存字节流。",
        )
        git_repo_path = st.text_input(
            "代码目录路径",
            value=st.session_state.get("sidebar_git_repo_path", ""),
            placeholder=r"D:\projects\demo-repo",
            help="既支持 Git 仓库，也支持从 Zip 解压出来的普通源码目录。",
        ).strip()

        st.markdown("## 运行配置")
        api_key = (
            st.text_input(
                "API Key",
                value=st.session_state.get("sidebar_api_key", settings.api_key or ""),
                type="password",
                help="默认读取 `.env`，也可以在这里临时覆盖。",
            ).strip()
            or None
        )
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.get("sidebar_base_url", settings.base_url),
        ).strip()
        model_options = _build_model_options(settings.model_name)
        selected_model = st.selectbox(
            "模型选择",
            options=model_options,
            index=model_options.index(
                st.session_state.get("sidebar_model_name", settings.model_name)
                if st.session_state.get("sidebar_model_name", settings.model_name) in model_options
                else settings.model_name
            ),
        )

        st.session_state["sidebar_git_repo_path"] = git_repo_path
        st.session_state["sidebar_api_key"] = api_key or ""
        st.session_state["sidebar_base_url"] = base_url
        st.session_state["sidebar_model_name"] = selected_model

        pdf_bytes = uploaded_pdf.getvalue() if uploaded_pdf is not None else None
        pdf_name = uploaded_pdf.name if uploaded_pdf is not None else None
        return SidebarState(
            uploaded_pdf_name=pdf_name,
            uploaded_pdf_bytes=pdf_bytes,
            git_repo_path=git_repo_path,
            api_key=api_key,
            base_url=base_url,
            model_name=selected_model,
        )


def _build_model_options(current_model: str) -> tuple[str, ...]:
    """合并默认模型列表和当前环境模型。"""

    ordered_options = list(DEFAULT_MODEL_OPTIONS)
    if current_model not in ordered_options:
        ordered_options.insert(0, current_model)
    return tuple(dict.fromkeys(ordered_options))
