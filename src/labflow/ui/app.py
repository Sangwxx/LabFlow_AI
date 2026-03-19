"""LabFlow 知云版主界面。"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st

from labflow.config.settings import get_settings
from labflow.parsers.git_repo_parser import GitRepoParser, GitRepoParseResult
from labflow.parsers.pdf_parser import PDFParser, PDFParseResult
from labflow.reasoning.agent_executor import PlanAndExecuteAgent
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import AlignmentResult, CodeEvidence, PaperSection
from labflow.ui.pdf_viewer import render_pdf_viewer
from labflow.ui.sidebar import SidebarState, render_sidebar

EVIDENCE_BUILDER = EvidenceBuilder()


@dataclass(frozen=True)
class WorkspaceState:
    """工作区解析状态。"""

    pdf_bytes: bytes | None
    pdf_name: str | None
    pdf_result: PDFParseResult | None
    pdf_error: str | None
    repo_result: GitRepoParseResult | None
    repo_error: str | None
    focus_sections: tuple[PaperSection, ...]
    project_structure: str


def run() -> None:
    """运行主界面。"""

    settings = get_settings()
    st.set_page_config(
        page_title="LabFlow 知云版",
        page_icon="LF",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    inject_styles()

    sidebar_state = render_sidebar(settings)
    sync_sidebar_overrides(sidebar_state)

    screen = st.empty()
    with screen.container():
        if st.session_state["current_route"] == "workspace":
            render_workspace()
        else:
            render_landing()


def init_session_state() -> None:
    """初始化会话状态。"""

    st.session_state.setdefault("current_route", "landing")
    st.session_state.setdefault("landing_pdf_bytes", None)
    st.session_state.setdefault("landing_pdf_name", None)
    st.session_state.setdefault("landing_git_repo_path", "")
    st.session_state.setdefault("workspace_signature", None)
    st.session_state.setdefault("workspace_data", None)
    st.session_state.setdefault("selected_section_index", 0)
    st.session_state.setdefault("pdf_hotspot_viewer", None)
    st.session_state.setdefault("semantic_alignment_cache", {})


@st.cache_resource(show_spinner=False)
def get_alignment_agent() -> PlanAndExecuteAgent:
    """把 Agent 执行器缓存起来，避免每次点击都重建客户端。"""

    return PlanAndExecuteAgent()


def render_landing() -> None:
    """渲染极简门户入口。"""

    st.markdown(
        """
        <div class="landing-shell">
            <div class="landing-kicker">LabFlow 知云版</div>
            <div class="landing-title">先进入工作区，再开始阅读与对齐</div>
            <div class="landing-body">
                这里只做两件事：接住你的论文 PDF，接住你的本地代码路径。
                进入工作区后，主界面会彻底切换为沉浸式阅读视图。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_column, git_column = st.columns(2, gap="large")
    with pdf_column:
        st.markdown('<div class="entry-card">', unsafe_allow_html=True)
        st.markdown("### 上传论文 PDF")
        st.caption("支持直接读取浏览器上传后的内存字节流。")
        uploaded_pdf = st.file_uploader(
            "选择 PDF",
            type=["pdf"],
            key="landing_pdf_uploader",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with git_column:
        st.markdown('<div class="entry-card">', unsafe_allow_html=True)
        st.markdown("### 输入代码路径")
        st.caption("支持真实 Git 仓库，也支持 GitHub Zip 解压后的普通 Python 目录。")
        git_repo_path = st.text_area(
            "代码路径",
            value=st.session_state.get("landing_git_repo_path", ""),
            placeholder=r"E:\project\your-repo",
            height=190,
            key="landing_git_repo_path_input",
            label_visibility="collapsed",
        ).strip()
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_pdf is not None:
        st.session_state["landing_pdf_bytes"] = uploaded_pdf.getvalue()
        st.session_state["landing_pdf_name"] = uploaded_pdf.name
    st.session_state["landing_git_repo_path"] = git_repo_path

    center_left, center_mid, center_right = st.columns([1, 1.35, 1])
    with center_mid:
        if st.button("进入工作区", type="primary", use_container_width=True):
            if not st.session_state.get("landing_pdf_bytes"):
                st.warning("先上传论文 PDF，再进入工作区。")
                return
            if not st.session_state.get("landing_git_repo_path"):
                st.warning("先填写本地代码路径，再进入工作区。")
                return

            st.session_state["current_route"] = "workspace"
            st.session_state["selected_section_index"] = 0
            st.rerun()


def render_workspace() -> None:
    """渲染沉浸式阅读工作区。"""

    workspace = get_workspace_state()
    render_workspace_header()

    if workspace.pdf_bytes is None:
        render_workspace_empty_state()
        return

    left_column, right_column = st.columns([1.25, 1.15], gap="small")
    with left_column:
        selected_section = render_pdf_panel(workspace)
    with right_column:
        render_code_panel(workspace, selected_section)


def render_workspace_header() -> None:
    """渲染工作区顶部，我会把工具栏压到最薄，尽量把首屏还给内容。"""

    header_left, header_mid, header_right = st.columns([5, 2.2, 1.2], gap="small")
    with header_left:
        st.markdown(
            """
            <div class="workspace-bar">
                <div class="workspace-title">阅读工作区</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_mid:
        st.caption("左侧连续阅读论文，右侧保持代码命中与即时结论。")
    with header_right:
        if st.button("返回首页", use_container_width=True):
            st.session_state["current_route"] = "landing"
            st.rerun()


def render_workspace_empty_state() -> None:
    """工作区空态。"""

    st.info("当前还没有可展示的工作区内容。先回到首页完成 PDF 和代码路径输入。")


def get_workspace_state() -> WorkspaceState:
    """根据当前输入准备工作区数据。"""

    pdf_bytes, pdf_name = resolve_pdf_source()
    git_repo_path = resolve_git_repo_path()
    signature = build_workspace_signature(pdf_bytes, pdf_name, git_repo_path)

    if (
        st.session_state.get("workspace_signature") == signature
        and st.session_state.get("workspace_data") is not None
    ):
        return st.session_state["workspace_data"]

    progress = st.progress(0, text="正在准备工作区...")
    progress.progress(15, text="正在接入 PDF 输入...")

    pdf_result: PDFParseResult | None = None
    pdf_error: str | None = None
    repo_result: GitRepoParseResult | None = None
    repo_error: str | None = None
    focus_sections: tuple[PaperSection, ...] = ()
    project_structure = "当前代码目录为空。"

    if pdf_bytes is not None:
        try:
            pdf_result = load_pdf_result(pdf_bytes, pdf_name or "uploaded.pdf")
            focus_sections = load_focus_sections(pdf_result)
        except (RuntimeError, ValueError) as exc:
            pdf_error = str(exc)

    progress.progress(52, text="正在接入代码目录...")
    if git_repo_path:
        try:
            repo_result = load_repo_result(git_repo_path)
            project_structure = load_project_structure(repo_result)
        except (FileNotFoundError, ValueError) as exc:
            repo_error = str(exc)

    progress.progress(100, text="工作区准备完成")
    progress.empty()

    workspace = WorkspaceState(
        pdf_bytes=pdf_bytes,
        pdf_name=pdf_name,
        pdf_result=pdf_result,
        pdf_error=pdf_error,
        repo_result=repo_result,
        repo_error=repo_error,
        focus_sections=focus_sections,
        project_structure=project_structure,
    )
    st.session_state["workspace_signature"] = signature
    st.session_state["workspace_data"] = workspace
    if st.session_state.get("semantic_alignment_cache_signature") != signature:
        st.session_state["semantic_alignment_cache_signature"] = signature
        st.session_state["semantic_alignment_cache"] = {}
    sync_section_selection(workspace)
    return workspace


def resolve_pdf_source() -> tuple[bytes | None, str | None]:
    """优先使用工作区侧边栏新上传的 PDF，否则回退到首页入口文件。"""

    sidebar_bytes = st.session_state.get("sidebar_uploaded_pdf_bytes")
    sidebar_name = st.session_state.get("sidebar_uploaded_pdf_name")
    if sidebar_bytes:
        return sidebar_bytes, sidebar_name or "uploaded.pdf"
    return (
        st.session_state.get("landing_pdf_bytes"),
        st.session_state.get("landing_pdf_name"),
    )


def resolve_git_repo_path() -> str:
    """优先使用侧边栏路径，否则回退到首页输入。"""

    return (
        st.session_state.get("sidebar_git_repo_path")
        or st.session_state.get("landing_git_repo_path", "")
    ).strip()


def build_workspace_signature(
    pdf_bytes: bytes | None,
    pdf_name: str | None,
    git_repo_path: str,
) -> str:
    """生成工作区缓存签名。"""

    pdf_size = len(pdf_bytes) if pdf_bytes else 0
    return f"{pdf_name or 'none'}::{pdf_size}::{git_repo_path}"


def sync_section_selection(workspace: WorkspaceState) -> None:
    """当章节列表变化时重置选中态。"""

    section_count = len(workspace.focus_sections)
    if section_count == 0:
        st.session_state["selected_section_index"] = 0
        return

    current_index = st.session_state.get("selected_section_index", 0)
    if current_index >= section_count:
        st.session_state["selected_section_index"] = 0


@st.cache_data(show_spinner=False)
def load_pdf_result(pdf_bytes: bytes, source_name: str) -> PDFParseResult:
    """缓存 PDF 解析结果。"""

    return PDFParser().parse_stream(pdf_bytes, source_name=source_name)


@st.cache_data(show_spinner=False)
def load_repo_result(repo_path: str) -> GitRepoParseResult:
    """缓存代码目录解析结果。"""

    return GitRepoParser().parse(repo_path)


@st.cache_data(show_spinner=False)
def load_focus_sections(pdf_result: PDFParseResult) -> tuple[PaperSection, ...]:
    """缓存段落级阅读焦点结果。"""

    return EVIDENCE_BUILDER.build_focus_sections(pdf_result)


@st.cache_data(show_spinner=False)
def load_code_evidences(repo_result: GitRepoParseResult) -> tuple[CodeEvidence, ...]:
    """只在用户点击论文片段后再懒加载代码证据。"""

    return EVIDENCE_BUILDER.build_code_evidences(repo_result)


@st.cache_data(show_spinner=False)
def load_project_structure(repo_result: GitRepoParseResult) -> str:
    """初始化阶段只构建文件树，确保工作区秒开。"""

    return EVIDENCE_BUILDER.build_project_structure(repo_result)


def render_pdf_panel(workspace: WorkspaceState) -> PaperSection | None:
    """渲染左侧论文阅读区。"""

    if workspace.pdf_error:
        st.error(workspace.pdf_error)
        if "PyMuPDF" in workspace.pdf_error:
            st.caption("请先执行 `python -m pip install pymupdf`，然后刷新页面。")

    sync_hotspot_selection(workspace)
    selected_section = render_section_picker(workspace.focus_sections)
    if selected_section is not None:
        st.caption(f"P{selected_section.page_number} · {selected_section.title}")

    if workspace.pdf_bytes is not None:
        try:
            render_pdf_viewer(
                workspace.pdf_bytes,
                blocks=workspace.pdf_result.blocks if workspace.pdf_result is not None else (),
                height=1180,
                page_number=selected_section.page_number if selected_section else None,
                selected_block_order=selected_section.order if selected_section else None,
                key="pdf_hotspot_viewer",
            )
        except RuntimeError as exc:
            st.error(str(exc))
    return selected_section


def render_section_picker(paper_sections: tuple[PaperSection, ...]) -> PaperSection | None:
    """渲染章节定位器。"""

    if not paper_sections:
        st.info("当前 PDF 还没有抽取出可点击的章节。")
        return None

    selected_index = st.selectbox(
        "章节定位",
        options=tuple(range(len(paper_sections))),
        index=min(st.session_state.get("selected_section_index", 0), len(paper_sections) - 1),
        format_func=lambda index: format_section_label(paper_sections[index]),
        key="section_picker",
        label_visibility="collapsed",
    )
    st.session_state["selected_section_index"] = selected_index
    return paper_sections[selected_index]


def format_section_label(section: PaperSection) -> str:
    """格式化段落标签。"""

    preview = section.content.replace("\n", " ").strip()
    if len(preview) > 42:
        preview = preview[:42] + "..."
    return f"P{section.page_number} · {section.title} · {preview}"


def sync_hotspot_selection(workspace: WorkspaceState) -> None:
    """把 PDF 热区组件返回的段落编号同步回当前选中态。"""

    if not workspace.focus_sections:
        return

    focus_block = st.session_state.get("pdf_hotspot_viewer")
    if focus_block is None:
        return

    try:
        block_order = int(focus_block)
    except (TypeError, ValueError):
        return

    for index, section in enumerate(workspace.focus_sections):
        if section.order == block_order:
            st.session_state["selected_section_index"] = index
            st.session_state["applied_focus_block"] = str(focus_block)
            break


def render_code_panel(
    workspace: WorkspaceState,
    selected_section: PaperSection | None,
) -> None:
    """渲染右侧代码面板，让 Agent 先规划、再执行、后总结。"""

    if workspace.repo_error:
        st.error(workspace.repo_error)
        return

    if workspace.repo_result is None:
        st.info("先准备代码目录，右侧才会显示对应代码。")
        return

    if selected_section is None:
        st.info("先在左侧选择一个章节。")
        return

    if (
        not workspace.repo_result.source_files
        and not workspace.repo_result.working_tree_diff.strip()
    ):
        st.warning("当前目录下没有可用于联动展示的 Python 代码。")
        return

    status_panel = st.status("Agent 正在规划与执行", expanded=True)

    def handle_agent_event(event: dict) -> None:
        kind = str(event.get("kind", "")).strip()
        if kind == "plan_update":
            remaining_steps = event.get("remaining_steps", ())
            if remaining_steps:
                status_panel.write("**[Current Plan]**")
                for step_text in remaining_steps:
                    status_panel.write(f"- {step_text}")
            if event.get("message"):
                status_panel.write(f"**[Thought]** {event['message']}")
            return
        if kind == "current_plan":
            status_panel.write(f"**[Current Plan]** {event.get('message', '')}")
            return
        if kind == "thought":
            status_panel.write(f"**[Thought]** {event.get('message', '')}")
            return
        if kind == "action":
            action_input = event.get("action_input", {})
            status_panel.write(f"**[Action]** {event.get('message', '')} | 输入: `{action_input}`")
            return
        if kind == "observation":
            status_panel.write(f"**[Observation]** {event.get('message', '')}")
            return
        if kind == "cache_hit":
            status_panel.write(f"**[Current Plan]** {event.get('message', '')}")

    with st.spinner("正在让 Agent 先思考、再拆解、后执行，请稍等..."):
        alignment_result = get_semantic_alignment(
            workspace_signature=st.session_state.get("workspace_signature", ""),
            selected_section=selected_section,
            repo_result=workspace.repo_result,
            project_structure=workspace.project_structure,
            event_handler=handle_agent_event,
        )

    if alignment_result is None:
        status_panel.update(label="Agent 未找到可解释实现", state="error")
        st.warning("当前章节暂时没有找到可解释的代码实现。")
        return

    status_panel.update(label="Agent 执行完成", state="complete")
    render_code_canvas(alignment_result)


def get_semantic_alignment(
    *,
    workspace_signature: str,
    selected_section: PaperSection,
    repo_result: GitRepoParseResult,
    project_structure: str,
    event_handler=None,
) -> AlignmentResult | None:
    """缓存单段语义对齐结果，避免同一段落反复触发远端推理。"""

    cache_key = f"{workspace_signature}:{selected_section.order}"
    cached_result = st.session_state["semantic_alignment_cache"].get(cache_key)
    if cached_result is not None:
        if event_handler is not None:
            event_handler({"kind": "cache_hit", "message": "命中缓存，直接复用上次 Agent 结果。"})
        return cached_result

    code_evidences = load_code_evidences(repo_result)
    if not code_evidences:
        return None

    alignment_result = get_alignment_agent().run(
        selected_section,
        code_evidences,
        project_structure=project_structure,
        event_handler=event_handler,
    )
    st.session_state["semantic_alignment_cache"][cache_key] = alignment_result
    return alignment_result


def render_code_canvas(alignment_result: AlignmentResult) -> None:
    """右栏优先展示 Agent 的检索计划、实现链路和自我审计，再用高亮代码承接结论。"""

    verdict_label = {
        "strong_match": "语义强匹配",
        "partial_match": "部分实现",
        "missing_implementation": "实现缺口",
        "formula_mismatch": "公式/参数偏离",
    }.get(alignment_result.match_type, "语义判断")
    st.info(
        " · ".join(
            [
                f"{verdict_label} {alignment_result.score_out_of_ten:.1f}/10",
                alignment_result.code_file_name,
                f"L{alignment_result.code_start_line}-L{alignment_result.code_end_line}",
            ]
        )
    )
    if alignment_result.needs_manual_review:
        st.warning(
            alignment_result.confidence_note
            or "我找到了相关代码，但在变量映射上存在歧义，建议人工核对。"
        )
    elif alignment_result.confidence_note:
        st.info(alignment_result.confidence_note)

    if alignment_result.retrieval_plan:
        st.markdown("### Agent 检索计划")
        st.markdown(alignment_result.retrieval_plan)

    st.markdown("### 实现链路分析")
    st.markdown(alignment_result.implementation_chain or alignment_result.analysis)
    st.markdown("### Agent 理解证据")
    st.markdown(alignment_result.semantic_evidence)
    if alignment_result.reflection:
        st.markdown("### 自我审计")
        st.markdown(alignment_result.reflection)
    if alignment_result.step_traces:
        with st.expander("查看 Plan-and-Execute 执行轨迹", expanded=False):
            for trace in alignment_result.step_traces:
                st.markdown(f"**[Current Plan]** {trace.step.display_text}")
                st.markdown(f"**[Thought]** {trace.thought}")
                st.markdown(f"**[Action]** {trace.action}")
                st.markdown(f"**[Observation]** {trace.observation}")
                for invocation in trace.tool_invocations:
                    st.markdown(f"- `{invocation.tool_name}` | 输入: `{invocation.tool_input}`")
    with st.container(height=1120):
        st.markdown(
            build_highlighted_code_html(alignment_result),
            unsafe_allow_html=True,
        )


def build_highlighted_code_html(alignment_result: AlignmentResult) -> str:
    """把模型挑出来的关键逻辑行高亮出来，降低用户二次定位成本。"""

    highlighted_lines = set(alignment_result.highlighted_line_numbers)
    html_lines: list[str] = []
    snippet_lines = alignment_result.code_snippet.splitlines() or [""]
    for offset, raw_line in enumerate(snippet_lines):
        absolute_line = alignment_result.code_start_line + offset
        is_highlighted = absolute_line in highlighted_lines
        line_class = "code-line code-line-highlight" if is_highlighted else "code-line"
        html_lines.append(
            f"""
            <div class="{line_class}">
                <span class="code-line-number">{absolute_line}</span>
                <span class="code-line-content">{escape(raw_line) or "&nbsp;"}</span>
            </div>
            """
        )

    return f"""
    <div class="semantic-code-shell">
        <div class="semantic-code-header">{alignment_result.code_file_name}</div>
        <div class="semantic-code-body">
            {"".join(html_lines)}
        </div>
    </div>
    """


def sync_sidebar_overrides(sidebar_state: SidebarState) -> None:
    """把侧边栏中的输入覆盖到工作区缓存源。"""

    if sidebar_state.uploaded_pdf_bytes:
        st.session_state["sidebar_uploaded_pdf_bytes"] = sidebar_state.uploaded_pdf_bytes
        st.session_state["sidebar_uploaded_pdf_name"] = sidebar_state.uploaded_pdf_name
    if sidebar_state.git_repo_path:
        st.session_state["sidebar_git_repo_path"] = sidebar_state.git_repo_path


def inject_styles() -> None:
    """注入沉浸式样式。"""

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+SC:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: "IBM Plex Sans SC", "Microsoft YaHei", sans-serif;
            }

            #MainMenu,
            footer,
            header[data-testid="stHeader"],
            div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0;
            }

            .stApp {
                background:
                    radial-gradient(
                        circle at top right,
                        rgba(255, 201, 142, 0.16),
                        transparent 24%
                    ),
                    radial-gradient(
                        circle at left 28%,
                        rgba(38, 147, 125, 0.12),
                        transparent 22%
                    ),
                    linear-gradient(180deg, #f8f4ec 0%, #f1ece2 100%);
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 98vw;
            }

            .stCodeBlock {
                line-height: 1.2;
            }

            .stCodeBlock pre {
                line-height: 1.2;
            }

            div[data-testid="column"] {
                width: 100% !important;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(19, 31, 49, 0.98), rgba(35, 57, 76, 0.97));
            }

            section[data-testid="stSidebar"] * {
                color: #f9f3e6;
            }

            .landing-shell {
                max-width: 60rem;
                margin: 6rem auto 2rem auto;
                text-align: center;
            }

            .landing-kicker {
                font-size: 0.88rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #8f7358;
                margin-bottom: 0.8rem;
            }

            .landing-title {
                font-size: 3rem;
                line-height: 1.1;
                font-weight: 700;
                color: #1a2b3d;
                margin-bottom: 0.9rem;
            }

            .landing-body {
                font-size: 1.05rem;
                line-height: 1.9;
                color: #415668;
            }

            .entry-card {
                border-radius: 26px;
                padding: 1.15rem 1.2rem;
                background: rgba(255, 252, 247, 0.92);
                border: 1px solid rgba(20, 37, 58, 0.08);
                box-shadow: 0 18px 42px rgba(20, 37, 58, 0.06);
                min-height: 16rem;
            }

            .entry-card {
                min-height: 22rem;
            }

            .workspace-bar {
                margin-bottom: 0.3rem;
            }

            .workspace-title {
                font-size: 1.2rem;
                font-weight: 700;
                color: #1a2b3d;
                margin-bottom: 0rem;
            }

            .stButton > button {
                min-height: 3.25rem;
                font-size: 1.02rem;
                font-weight: 700;
                border-radius: 999px;
                border: 0;
                box-shadow: 0 14px 32px rgba(20, 37, 58, 0.10);
            }

            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
                width: 100%;
            }

            div[data-testid="stAlert"] {
                margin-bottom: 0.35rem;
            }

            .semantic-code-shell {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(20, 37, 58, 0.08);
                background: rgba(248, 250, 252, 0.96);
                box-shadow: 0 14px 30px rgba(20, 37, 58, 0.06);
            }

            .semantic-code-header {
                padding: 0.8rem 1rem;
                font-size: 0.92rem;
                font-weight: 700;
                color: #1e3a5f;
                background: rgba(226, 232, 240, 0.72);
                border-bottom: 1px solid rgba(20, 37, 58, 0.08);
            }

            .semantic-code-body {
                max-height: 1070px;
                overflow: auto;
                font-family: "Source Code Pro", "Consolas", monospace;
                font-size: 0.9rem;
                line-height: 1.45;
            }

            .code-line {
                display: grid;
                grid-template-columns: 4rem 1fr;
                gap: 0.8rem;
                padding: 0.16rem 0.9rem;
                border-left: 4px solid transparent;
                white-space: pre;
            }

            .code-line-highlight {
                background: rgba(255, 232, 184, 0.58);
                border-left-color: #d97706;
            }

            .code-line-number {
                color: #94a3b8;
                text-align: right;
                user-select: none;
            }

            .code-line-content {
                color: #0f172a;
                overflow-x: auto;
            }

            div[data-testid="stSelectbox"] {
                margin-bottom: 0.35rem;
            }

            iframe {
                min-height: 1180px;
            }

            textarea {
                font-size: 1rem !important;
                line-height: 1.7 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
