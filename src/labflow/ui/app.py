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
ALIGNMENT_CACHE_VERSION = "learning-output-v5"


@dataclass(frozen=True)
class WorkspaceState:
    pdf_bytes: bytes | None
    pdf_name: str | None
    pdf_result: PDFParseResult | None
    pdf_error: str | None
    repo_result: GitRepoParseResult | None
    repo_error: str | None
    focus_sections: tuple[PaperSection, ...]
    project_structure: str


def run() -> None:
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

    if st.session_state["current_route"] == "workspace":
        render_workspace()
    else:
        render_landing()


def init_session_state() -> None:
    st.session_state.setdefault("current_route", "landing")
    st.session_state.setdefault("landing_pdf_bytes", None)
    st.session_state.setdefault("landing_pdf_name", None)
    st.session_state.setdefault("landing_git_repo_path", "")
    st.session_state.setdefault("workspace_signature", None)
    st.session_state.setdefault("workspace_data", None)
    st.session_state.setdefault("selected_section_index", None)
    st.session_state.setdefault("pdf_hotspot_viewer", None)
    st.session_state.setdefault("semantic_alignment_cache", {})


@st.cache_resource(show_spinner=False)
def get_alignment_agent() -> PlanAndExecuteAgent:
    return PlanAndExecuteAgent()


def render_landing() -> None:
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

    _, middle, _ = st.columns([1, 1.35, 1])
    with middle:
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


def render_workspace() -> None:
    workspace = get_workspace_state()
    render_workspace_header()

    if workspace.pdf_bytes is None:
        st.info("当前还没有可展示的工作区内容。先回到首页完成 PDF 和代码路径输入。")
        return

    left_column, right_column = st.columns([1.25, 1.15], gap="small")
    with left_column:
        selected_section = render_pdf_panel(workspace)
    with right_column:
        render_code_panel(workspace, selected_section)


def render_workspace_header() -> None:
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
        st.caption("左侧连续阅读论文，右侧只保留代码命中与导师式解释。")
    with header_right:
        if st.button("返回首页", use_container_width=True):
            st.session_state["current_route"] = "landing"
            st.rerun()


def get_workspace_state() -> WorkspaceState:
    pdf_bytes, pdf_name = resolve_pdf_source()
    git_repo_path = resolve_git_repo_path()
    signature = build_workspace_signature(pdf_bytes, pdf_name, git_repo_path)

    if (
        st.session_state.get("workspace_signature") == signature
        and st.session_state.get("workspace_data") is not None
    ):
        return st.session_state["workspace_data"]

    pdf_result = None
    pdf_error = None
    repo_result = None
    repo_error = None
    focus_sections: tuple[PaperSection, ...] = ()
    project_structure = "当前代码目录为空。"

    if pdf_bytes is not None:
        try:
            pdf_result = load_pdf_result(pdf_bytes, pdf_name or "uploaded.pdf")
            focus_sections = load_focus_sections(pdf_result)
        except (RuntimeError, ValueError) as exc:
            pdf_error = str(exc)

    if git_repo_path:
        try:
            repo_result = load_repo_result(git_repo_path)
            project_structure = load_project_structure(repo_result)
        except (FileNotFoundError, ValueError) as exc:
            repo_error = str(exc)

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
    sidebar_bytes = st.session_state.get("sidebar_uploaded_pdf_bytes")
    sidebar_name = st.session_state.get("sidebar_uploaded_pdf_name")
    if sidebar_bytes:
        return sidebar_bytes, sidebar_name or "uploaded.pdf"
    return st.session_state.get("landing_pdf_bytes"), st.session_state.get("landing_pdf_name")


def resolve_git_repo_path() -> str:
    return (
        st.session_state.get("sidebar_git_repo_path")
        or st.session_state.get("landing_git_repo_path", "")
    ).strip()


def build_workspace_signature(
    pdf_bytes: bytes | None,
    pdf_name: str | None,
    git_repo_path: str,
) -> str:
    pdf_size = len(pdf_bytes) if pdf_bytes else 0
    return f"{pdf_name or 'none'}::{pdf_size}::{git_repo_path}"


def sync_section_selection(workspace: WorkspaceState) -> None:
    if not workspace.focus_sections:
        st.session_state["selected_section_index"] = None
        return
    current_index = st.session_state.get("selected_section_index")
    if current_index is None:
        return
    if current_index >= len(workspace.focus_sections):
        st.session_state["selected_section_index"] = None


@st.cache_data(show_spinner=False)
def load_pdf_result(pdf_bytes: bytes, source_name: str) -> PDFParseResult:
    return PDFParser().parse_stream(pdf_bytes, source_name=source_name)


@st.cache_data(show_spinner=False)
def load_repo_result(repo_path: str) -> GitRepoParseResult:
    return GitRepoParser().parse(repo_path)


@st.cache_data(show_spinner=False)
def load_focus_sections(pdf_result: PDFParseResult) -> tuple[PaperSection, ...]:
    return EVIDENCE_BUILDER.build_focus_sections(pdf_result)


@st.cache_data(show_spinner=False)
def load_code_evidences(repo_result: GitRepoParseResult) -> tuple[CodeEvidence, ...]:
    return EVIDENCE_BUILDER.build_code_evidences(repo_result)


@st.cache_data(show_spinner=False)
def load_project_structure(repo_result: GitRepoParseResult) -> str:
    return EVIDENCE_BUILDER.build_project_structure(repo_result)


def render_pdf_panel(workspace: WorkspaceState) -> PaperSection | None:
    if workspace.pdf_error:
        st.error(workspace.pdf_error)
        if "PyMuPDF" in workspace.pdf_error:
            st.caption("请先执行 `python -m pip install pymupdf`，然后刷新页面。")

    sync_hotspot_selection(workspace)
    selected_section = get_selected_section(workspace.focus_sections)
    render_section_focus_bar(selected_section)

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


def get_selected_section(paper_sections: tuple[PaperSection, ...]) -> PaperSection | None:
    if not paper_sections:
        st.info("当前 PDF 还没有抽取出可点击的章节。")
        return None

    current_index = st.session_state.get("selected_section_index")
    if not isinstance(current_index, int):
        return None
    if not 0 <= current_index < len(paper_sections):
        return None
    return paper_sections[current_index]


def render_section_focus_bar(section: PaperSection | None) -> None:
    if section is None:
        st.caption("点击左侧论文中的段落后，这里会显示当前聚焦片段。")
        return
    st.markdown(
        (
            '<div class="section-focus-bar">'
            f'<span class="section-focus-page">P{section.page_number}</span>'
            f'<span class="section-focus-title">{section.title}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def format_section_label(section: PaperSection) -> str:
    preview = section.content.replace("\n", " ").strip()
    if len(preview) > 42:
        preview = preview[:42] + "..."
    return f"P{section.page_number} · {section.title} · {preview}"


def sync_hotspot_selection(workspace: WorkspaceState) -> None:
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
            break


def render_code_panel(workspace: WorkspaceState, selected_section: PaperSection | None) -> None:
    if workspace.repo_error:
        st.error(workspace.repo_error)
        return
    if workspace.repo_result is None:
        st.info("先准备代码目录，右侧才会显示对应代码。")
        return
    if selected_section is None:
        st.info("请选择论文片段开始深度对齐。")
        return

    trace_events: list[dict] = []
    trace_placeholder = st.empty()

    def handle_agent_event(event: dict) -> None:
        trace_events.append(event)
        render_trace_panel(trace_placeholder, trace_events, finalized=False)

    try:
        with st.spinner("Agent 正在先思考、再拆解、后执行，请稍等..."):
            alignment_result = get_semantic_alignment(
                workspace_signature=st.session_state.get("workspace_signature", ""),
                selected_section=selected_section,
                repo_result=workspace.repo_result,
                project_structure=workspace.project_structure,
                event_handler=handle_agent_event,
            )
    except Exception as exc:  # noqa: BLE001
        trace_placeholder.empty()
        st.error(f"Agent 推理过程中出现异常：{exc}")
        st.info("本轮已停止推理。建议稍后重试，或先把注意力放回论文片段本身。")
        return

    render_trace_panel(trace_placeholder, trace_events, finalized=True)
    if alignment_result is None:
        st.warning("本轮模型没有稳定返回，我已停止展示中间推理链。")
        return
    render_code_canvas(alignment_result)


def get_semantic_alignment(
    *,
    workspace_signature: str,
    selected_section: PaperSection,
    repo_result: GitRepoParseResult,
    project_structure: str,
    event_handler=None,
) -> AlignmentResult | None:
    cache_key = f"{ALIGNMENT_CACHE_VERSION}:{workspace_signature}:{selected_section.order}"
    cached_result = st.session_state["semantic_alignment_cache"].get(cache_key)
    if cached_result is not None:
        if event_handler is not None:
            event_handler({"kind": "cache_hit", "message": "命中缓存，直接复用上一次 Agent 结果。"})
        return cached_result

    code_evidences = load_code_evidences(repo_result)
    alignment_result = get_alignment_agent().run(
        selected_section,
        code_evidences,
        project_structure=project_structure,
        event_handler=event_handler,
    )
    st.session_state["semantic_alignment_cache"][cache_key] = alignment_result
    return alignment_result


def render_trace_panel(placeholder, trace_events: list[dict], *, finalized: bool) -> None:
    if not trace_events or finalized:
        placeholder.empty()
        return

    with placeholder.container():
        with st.container(height=300):
            status = st.status("Agent 正在推理...", expanded=True)
            for event in trace_events:
                render_trace_event(event, status)


def render_trace_event(event: dict, status=None) -> None:
    writer = status.write if status is not None else st.write
    kind = str(event.get("kind", "")).strip()
    if kind == "plan_update":
        remaining_steps = event.get("remaining_steps", ())
        if remaining_steps:
            writer("**[Current Plan]**")
            for step_text in remaining_steps:
                writer(f"- {step_text}")
        if event.get("message"):
            writer(f"**[Thought]** {event['message']}")
        return
    if kind == "current_plan":
        writer(f"**[Current Plan]** {event.get('message', '')}")
        return
    if kind == "thought":
        writer(f"**[Thought]** {event.get('message', '')}")
        return
    if kind == "action":
        writer(f"**[Action]** {event.get('message', '')} | 输入: `{event.get('action_input', {})}`")
        return
    if kind == "observation":
        writer(f"**[Observation]** {event.get('message', '')}")
        return
    if kind == "cache_hit":
        writer(f"**[Current Plan]** {event.get('message', '')}")


def render_code_canvas(alignment_result: AlignmentResult) -> None:
    st.markdown("### 【中文译文】")
    st.markdown(alignment_result.analysis)

    st.markdown("### 【核心要点】")
    st.markdown(alignment_result.semantic_evidence or "当前暂无重点提炼。")

    st.markdown("### 【术语百科】")
    st.markdown(alignment_result.research_supplement or "这一段没有特别需要额外展开的术语。")

    if should_render_source_grounding(alignment_result):
        st.markdown("### 【源码落地】")
        st.caption(
            f"{alignment_result.code_file_name} · "
            f"L{alignment_result.code_start_line}-L{alignment_result.code_end_line}"
        )
        st.markdown(alignment_result.implementation_chain)
        with st.container(height=1120):
            st.markdown(build_highlighted_code_html(alignment_result), unsafe_allow_html=True)


def should_render_source_grounding(alignment_result: AlignmentResult) -> bool:
    return (
        alignment_result.match_type == "strong_match"
        and alignment_result.alignment_score >= 0.78
        and bool(alignment_result.implementation_chain.strip())
        and not alignment_result.code_snippet.startswith("# 当前未定位到对应源码")
    )


def build_highlighted_code_html(alignment_result: AlignmentResult) -> str:
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
    if sidebar_state.uploaded_pdf_bytes:
        st.session_state["sidebar_uploaded_pdf_bytes"] = sidebar_state.uploaded_pdf_bytes
        st.session_state["sidebar_uploaded_pdf_name"] = sidebar_state.uploaded_pdf_name
    if sidebar_state.git_repo_path:
        st.session_state["sidebar_git_repo_path"] = sidebar_state.git_repo_path


def inject_styles() -> None:
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

            div[data-testid="column"] {
                width: 100% !important;
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
                margin-bottom: 2rem;
            }

            .entry-card {
                min-height: 15rem;
                padding: 1.4rem;
                border-radius: 24px;
                background: rgba(255, 255, 255, 0.78);
                box-shadow: 0 20px 40px rgba(28, 46, 66, 0.10);
            }

            .workspace-title {
                font-size: 2.4rem;
                font-weight: 700;
                color: #1a2b3d;
            }

            .section-focus-bar {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                min-height: 3.25rem;
                padding: 0.75rem 0.95rem;
                margin: 0 0 0.65rem 0;
                border-radius: 18px;
                background: rgba(239, 241, 246, 0.92);
                border: 1px solid rgba(34, 47, 62, 0.08);
            }

            .section-focus-page {
                flex: 0 0 auto;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                background: rgba(31, 43, 61, 0.08);
                color: #1a2b3d;
                font-weight: 600;
                font-size: 0.92rem;
            }

            .section-focus-title {
                color: #2f4052;
                font-size: 1rem;
                font-weight: 500;
                line-height: 1.4;
            }

            .semantic-code-shell {
                border-radius: 22px;
                overflow: hidden;
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(196, 176, 154, 0.6);
            }

            .semantic-code-header {
                padding: 0.85rem 1rem;
                background: rgba(235, 241, 247, 0.92);
                color: #1f3550;
                font-weight: 600;
            }

            .semantic-code-body {
                padding: 0.75rem 0;
                overflow-x: auto;
                overflow-y: auto;
                white-space: pre;
            }

            .code-line {
                display: grid;
                grid-template-columns: 4.5rem minmax(0, 1fr);
                gap: 0.75rem;
                padding: 0.12rem 1rem;
                font-family: "Source Code Pro", "Consolas", monospace;
                font-size: 0.92rem;
                line-height: 1.2;
            }

            .code-line-highlight {
                background: rgba(255, 238, 186, 0.66);
            }

            .code-line-number {
                color: #8b97a6;
                text-align: right;
                user-select: none;
            }

            .code-line-content {
                color: #10253c;
                white-space: pre;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
