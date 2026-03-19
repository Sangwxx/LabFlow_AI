"""Streamlit 首页。"""

from __future__ import annotations

import streamlit as st

from labflow.config.settings import get_settings
from labflow.parsers.git_repo_parser import GitRepoParser, GitRepoParseResult
from labflow.parsers.pdf_parser import PDFParser, PDFParseResult
from labflow.ui.home_content import HomeContent, build_home_content
from labflow.ui.sidebar import SidebarState, render_sidebar


def render_stage_cards(content: HomeContent) -> None:
    """渲染四阶段能力卡片。"""

    columns = st.columns(len(content.stage_cards))
    for column, card in zip(columns, content.stage_cards, strict=False):
        column.markdown(
            f"""
            <div class="stage-card">
                <div class="stage-step">{card.step}</div>
                <div class="stage-title">{card.title}</div>
                <div class="stage-body">{card.description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_scope(title: str, items: tuple[str, ...]) -> None:
    """渲染列表型信息块。"""

    st.markdown(f"### {title}")
    for item in items:
        st.markdown(f"- {item}")


def render_parser_previews(sidebar_state: SidebarState) -> None:
    """根据侧边栏输入展示解析预览。"""

    st.markdown("## 感知预览")
    left_column, right_column = st.columns(2)

    with left_column:
        render_pdf_preview(sidebar_state)
    with right_column:
        render_git_preview(sidebar_state)


def render_pdf_preview(sidebar_state: SidebarState) -> None:
    """展示 PDF 解析结果。"""

    st.markdown("### PDF 解析")
    if not sidebar_state.uploaded_pdf_bytes:
        st.caption("上传 PDF 后，我会先把标题和正文块分开，方便后面的对齐推理。")
        return

    parser = PDFParser()
    try:
        result = parser.parse_bytes(
            pdf_bytes=sidebar_state.uploaded_pdf_bytes,
            source_name=sidebar_state.uploaded_pdf_name or "uploaded.pdf",
        )
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return

    st.success(f"已解析 {result.source_name}，共 {result.page_count} 页。")
    st.caption(
        f"识别出 {len(result.title_blocks)} 个标题块，{len(result.paragraph_blocks)} 个正文块。"
    )
    render_pdf_result(result)


def render_pdf_result(result: PDFParseResult) -> None:
    """渲染 PDF 解析结果摘要。"""

    with st.expander("标题预览", expanded=True):
        if not result.title_blocks:
            st.caption("当前没有命中明显标题，我会在后续迭代里继续调规则。")
        for block in result.title_blocks[:6]:
            st.markdown(f"- P{block.page_number} · {block.text}")

    with st.expander("正文字段预览", expanded=False):
        for block in result.paragraph_blocks[:3]:
            st.write(block.text)


def render_git_preview(sidebar_state: SidebarState) -> None:
    """展示 Git 仓库解析结果。"""

    st.markdown("### Git 仓库解析")
    if not sidebar_state.git_repo_path:
        st.caption("填入本地仓库路径后，我会拉出最近 10 次提交和当前工作区 diff。")
        return

    parser = GitRepoParser()
    try:
        result = parser.parse(sidebar_state.git_repo_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    st.success(f"已定位仓库：{result.repo_path}")
    st.caption(f"当前分支：{result.branch_name} · 最近提交数：{len(result.recent_commits)}")
    render_git_result(result)


def render_git_result(result: GitRepoParseResult) -> None:
    """渲染 Git 仓库解析结果摘要。"""

    with st.expander("最近 10 次提交", expanded=True):
        if not result.recent_commits:
            st.caption("这个仓库还没有提交记录。")
        for commit in result.recent_commits:
            st.markdown(f"- `{commit.short_sha}` {commit.summary} · {commit.author_name}")

    with st.expander("当前工作区 diff", expanded=False):
        if result.working_tree_diff:
            st.code(result.working_tree_diff, language="diff")
        else:
            st.caption("当前工作区没有未提交变更。")


def run() -> None:
    """运行首页界面。"""

    settings = get_settings()
    content = build_home_content(settings)

    st.set_page_config(
        page_title="LabFlow AI",
        page_icon="LF",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    sidebar_state = render_sidebar()

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+SC:wght@400;500;600;700&display=swap');

            html, body, [class*="css"]  {
                font-family: "IBM Plex Sans SC", "Microsoft YaHei", sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(
                        circle at top right,
                        rgba(255, 176, 120, 0.18),
                        transparent 28%
                    ),
                    radial-gradient(
                        circle at left 20%,
                        rgba(26, 166, 154, 0.16),
                        transparent 24%
                    ),
                    linear-gradient(180deg, #f7f3eb 0%, #f2eee6 100%);
            }

            .hero {
                padding: 1.8rem 2rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(20, 37, 58, 0.96), rgba(42, 80, 73, 0.92));
                color: #fff7eb;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 24px 60px rgba(18, 28, 42, 0.16);
                margin-bottom: 1rem;
            }

            .hero-kicker {
                font-size: 0.88rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #ffd7a8;
                margin-bottom: 0.75rem;
            }

            .hero-title {
                font-size: 2.9rem;
                font-weight: 700;
                line-height: 1.08;
                margin-bottom: 0.8rem;
            }

            .hero-subtitle {
                font-size: 1.05rem;
                line-height: 1.75;
                max-width: 58rem;
                color: rgba(255, 247, 235, 0.9);
            }

            .metric-card {
                border-radius: 20px;
                padding: 1rem 1.1rem;
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(20, 37, 58, 0.08);
                box-shadow: 0 10px 30px rgba(20, 37, 58, 0.06);
            }

            .metric-label {
                font-size: 0.85rem;
                color: #49606f;
                margin-bottom: 0.4rem;
            }

            .metric-value {
                font-size: 1.35rem;
                font-weight: 700;
                color: #14253a;
            }

            .stage-card {
                height: 100%;
                min-height: 220px;
                border-radius: 22px;
                padding: 1.15rem;
                background: rgba(255, 252, 246, 0.86);
                border: 1px solid rgba(20, 37, 58, 0.08);
                box-shadow: 0 10px 30px rgba(20, 37, 58, 0.06);
            }

            .stage-step {
                font-size: 0.82rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #0e7b71;
                margin-bottom: 0.8rem;
            }

            .stage-title {
                font-size: 1.22rem;
                font-weight: 700;
                color: #14253a;
                margin-bottom: 0.65rem;
            }

            .stage-body {
                font-size: 0.96rem;
                line-height: 1.75;
                color: #405261;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">HUST AI Competition / Stage 0</div>
            <div class="hero-title">{content.title}</div>
            <div class="hero-subtitle">{content.subtitle}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3)
    metrics = (
        ("当前重点", "论文-代码对齐分析"),
        ("Git 输入", "本地路径导入"),
        ("PDF 范围", "仅文本型 PDF"),
    )
    for column, metric in zip(metric_columns, metrics, strict=False):
        label, value = metric
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## 能力闭环")
    render_stage_cards(content)

    left_column, right_column = st.columns([1.1, 0.9])
    with left_column:
        render_scope("当前范围", content.current_scope)
    with right_column:
        render_scope("下一步实现", content.next_actions)

    render_parser_previews(sidebar_state)

    st.info(
        "阶段 1 的感知层输入已经接好。"
        " 接下来可以基于 PDF 结构块、提交历史和工作区 diff 继续推进对齐推理。"
    )
