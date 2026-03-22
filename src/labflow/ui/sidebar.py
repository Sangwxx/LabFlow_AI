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
SESSION_API_KEY_KEY = "sidebar_api_key"
SESSION_API_KEY_INPUT_KEY = "sidebar_api_key_input"


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
        api_key = _render_api_key_input(settings)
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
        st.session_state[SESSION_API_KEY_KEY] = api_key or ""
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


def _render_api_key_input(settings: Settings) -> str | None:
    """渲染不会回显真实密钥的输入框。"""

    stored_api_key = st.session_state.get(SESSION_API_KEY_KEY, "")
    status_text = _build_api_key_status(
        has_env_api_key=bool(settings.api_key),
        has_session_override=bool(stored_api_key),
    )
    if status_text:
        st.caption(status_text)

    st.text_input(
        "API Key",
        value="",
        key=SESSION_API_KEY_INPUT_KEY,
        type="password",
        placeholder="留空时继续使用 .env 中的 API Key",
        help="默认复用 `.env` 中的 API Key，出于安全原因这里不会回显真实值；如需临时覆盖，请在这里输入新值。",
        on_change=_apply_api_key_override,
    )
    if stored_api_key and st.button("清除会话覆盖", use_container_width=True):
        st.session_state[SESSION_API_KEY_KEY] = ""
        st.session_state[SESSION_API_KEY_INPUT_KEY] = ""
        st.rerun()
    return stored_api_key or None


def _apply_api_key_override() -> None:
    """把用户刚输入的密钥转存到会话态，并清空可见输入框。"""

    raw_value = st.session_state.get(SESSION_API_KEY_INPUT_KEY, "")
    api_key = raw_value.strip()
    if not api_key:
        return
    st.session_state[SESSION_API_KEY_KEY] = api_key
    st.session_state[SESSION_API_KEY_INPUT_KEY] = ""


def _build_api_key_status(*, has_env_api_key: bool, has_session_override: bool) -> str | None:
    """生成 API Key 输入框上方的状态提示。"""

    if has_session_override:
        return "当前会话已应用手动输入的 API Key，页面不会回显具体值。"
    if has_env_api_key:
        return "已检测到 `.env` 中的 API Key，当前输入框不会回显真实值。"
    return None
