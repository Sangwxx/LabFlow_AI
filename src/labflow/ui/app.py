"""Streamlit 首页。"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from labflow.config.settings import get_settings
from labflow.parsers.git_repo_parser import GitRepoParser, GitRepoParseResult
from labflow.parsers.pdf_parser import PDFParser, PDFParseResult
from labflow.reasoning.aligner import align, align_inputs
from labflow.reasoning.models import AlignmentResult, CodeEvidence, PaperSection
from labflow.ui.home_content import HomeContent, build_home_content
from labflow.ui.sidebar import SidebarState, render_sidebar


@dataclass(frozen=True)
class ParsedWorkspace:
    """当前页面解析出的输入状态。"""

    pdf_result: PDFParseResult | None
    pdf_error: str | None
    git_result: GitRepoParseResult | None
    git_error: str | None


class DemoMismatchLLMClient:
    """我先用一个可控假客户端把链路压通，联调时就不会被外部模型波动卡住。"""

    def generate_json(self, *, system_prompt: str, user_prompt: str, **_: object) -> dict:
        if "alpha = 0.70" in user_prompt and "alpha = 0.30" in user_prompt:
            return {
                "alignment_score": 0.18,
                "match_type": "formula_mismatch",
                "analysis": (
                    "论文章节要求损失权重 alpha=0.70、beta=0.30，"
                    "但代码片段里写成了 alpha=0.30、beta=0.70，"
                    "变量名一致但系数被对调，属于公式实现偏离。"
                ),
                "improvement_suggestion": (
                    "把 trainer.py 中的权重系数改回论文给出的比例，并补一条单测锁住参数顺序。"
                ),
            }

        return {
            "alignment_score": 0.62,
            "match_type": "partial_match",
            "analysis": "当前证据存在一定相关性，但还看不出完全闭环。",
            "improvement_suggestion": "继续补充更完整的代码片段和公式上下文。",
        }


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


def parse_sidebar_inputs(sidebar_state: SidebarState) -> ParsedWorkspace:
    """把侧边栏输入统一解析出来，避免同一轮渲染重复跑多次。"""

    pdf_result: PDFParseResult | None = None
    pdf_error: str | None = None
    git_result: GitRepoParseResult | None = None
    git_error: str | None = None

    if sidebar_state.uploaded_pdf_bytes:
        try:
            pdf_result = PDFParser().parse_bytes(
                pdf_bytes=sidebar_state.uploaded_pdf_bytes,
                source_name=sidebar_state.uploaded_pdf_name or "uploaded.pdf",
            )
        except (RuntimeError, ValueError) as exc:
            pdf_error = str(exc)

    if sidebar_state.git_repo_path:
        try:
            git_result = GitRepoParser().parse(sidebar_state.git_repo_path)
        except (FileNotFoundError, ValueError) as exc:
            git_error = str(exc)

    return ParsedWorkspace(
        pdf_result=pdf_result,
        pdf_error=pdf_error,
        git_result=git_result,
        git_error=git_error,
    )


def render_parser_previews(parsed_workspace: ParsedWorkspace) -> None:
    """根据侧边栏输入展示解析预览。"""

    st.markdown("## 感知预览")
    left_column, right_column = st.columns(2)

    with left_column:
        render_pdf_preview(parsed_workspace)
    with right_column:
        render_git_preview(parsed_workspace)

    render_reasoning_panel(parsed_workspace)


def render_pdf_preview(parsed_workspace: ParsedWorkspace) -> None:
    """展示 PDF 解析结果。"""

    st.markdown("### PDF 解析")
    if parsed_workspace.pdf_result is None and parsed_workspace.pdf_error is None:
        st.caption("上传 PDF 后，我会先把标题和正文块分开，方便后面的对齐推理。")
        return

    if parsed_workspace.pdf_error:
        st.error(parsed_workspace.pdf_error)
        return

    result = parsed_workspace.pdf_result
    if result is None:
        st.error("PDF 解析结果为空。")
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


def render_git_preview(parsed_workspace: ParsedWorkspace) -> None:
    """展示 Git 仓库解析结果。"""

    st.markdown("### Git 仓库解析")
    if parsed_workspace.git_result is None and parsed_workspace.git_error is None:
        st.caption("填入本地仓库路径后，我会拉出最近 10 次提交和当前工作区 diff。")
        return

    if parsed_workspace.git_error:
        st.error(parsed_workspace.git_error)
        return

    result = parsed_workspace.git_result
    if result is None:
        st.error("Git 仓库解析结果为空。")
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


def render_reasoning_panel(parsed_workspace: ParsedWorkspace) -> None:
    """渲染推理层入口和内置案例。"""

    st.markdown("## 推理预览")
    left_column, right_column = st.columns([1.1, 0.9])
    with left_column:
        render_live_alignment_panel(parsed_workspace)
    with right_column:
        render_demo_alignment_case()


def render_live_alignment_panel(parsed_workspace: ParsedWorkspace) -> None:
    """展示真实输入的对齐入口。"""

    st.markdown("### 真实输入对齐")
    if parsed_workspace.pdf_result is None or parsed_workspace.git_result is None:
        st.caption("把 PDF 和本地 Git 仓库都准备好后，我就能跑一次真实对齐。")
        return

    if st.button("运行对齐分析", type="primary", use_container_width=True):
        with st.spinner("我在压缩候选证据，并让模型做结构化判断..."):
            try:
                results = align(parsed_workspace.pdf_result, parsed_workspace.git_result)
            except RuntimeError as exc:
                st.error(str(exc))
                return
        render_alignment_results(results)


def render_demo_alignment_case() -> None:
    """展示一个参数错配案例，先把推理链路跑通。"""

    st.markdown("### 内置错配案例")
    st.caption("这组样例专门模拟“论文公式写的是一套权重，代码里却把参数写反了”的情况。")
    demo_results = align_inputs(
        paper_sections=(
            PaperSection(
                title="3.2 损失函数权重",
                content=(
                    "论文要求总损失写成 L = alpha * L_cls + beta * L_reg，"
                    "其中 alpha = 0.70，beta = 0.30。"
                ),
                level=2,
                page_number=3,
                order=5,
            ),
        ),
        code_evidences=(
            CodeEvidence(
                file_name="trainer.py",
                code_snippet=(
                    "alpha = 0.30\nbeta = 0.70\nloss = alpha * cls_loss + beta * reg_loss"
                ),
                related_git_diff=(
                    "@@ -10,3 +10,3 @@\n-alpha = 0.70\n+alpha = 0.30\n-beta = 0.30\n+beta = 0.70"
                ),
                symbols=("alpha", "beta", "loss", "cls_loss", "reg_loss"),
                commit_context=("fix: 调整训练权重",),
            ),
        ),
        llm_client=DemoMismatchLLMClient(),
        top_k=1,
    )
    render_alignment_results(demo_results)


def render_alignment_results(results: tuple[AlignmentResult, ...]) -> None:
    """渲染对齐结果。"""

    if not results:
        st.warning("当前没有召回到可分析的候选对。")
        return

    for result in results:
        score_label = f"{result.alignment_score:.2f}"
        title = f"{result.paper_section_title} ↔ {result.code_file_name}"
        st.markdown(
            f"""
            <div class="stage-card" style="min-height: 0; margin-bottom: 1rem;">
                <div class="stage-step">{result.match_type}</div>
                <div class="stage-title">{title}</div>
                <div class="stage-body">
                    <strong>对齐评分：</strong>{score_label}<br/>
                    <strong>分析结论：</strong>{result.analysis}<br/>
                    <strong>改进建议：</strong>{result.improvement_suggestion}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
    parsed_workspace = parse_sidebar_inputs(sidebar_state)

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
            <div class="hero-kicker">HUST AI Competition / Stage 2</div>
            <div class="hero-title">{content.title}</div>
            <div class="hero-subtitle">{content.subtitle}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3)
    metrics = (
        ("当前重点", "论文-代码对齐分析"),
        ("检索策略", "BM25 轻量召回"),
        ("推理模式", "结构化 JSON 输出"),
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

    render_parser_previews(parsed_workspace)

    st.info(
        "阶段 2 的推理层入口已经接好。"
        " 现在既可以用内置错配案例验证链路，也可以在模型依赖齐全时对真实输入发起分析。"
    )
