"""LabFlow 主界面。"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st

from labflow.clients.llm_client import LLMClient
from labflow.clients.semantic_scholar_client import SemanticScholarClient
from labflow.config.settings import Settings, get_settings
from labflow.parsers.git_repo_parser import GitRepoParser, GitRepoParseResult
from labflow.parsers.pdf_parser import PDFParser, PDFParseResult
from labflow.reasoning.agent_executor import PlanAndExecuteAgent
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import AlignmentResult, CodeEvidence, PaperSection
from labflow.reporting import ReadingNoteEntry, ReportGenerator
from labflow.ui.guide_page import render_quick_guide_page
from labflow.ui.landing import (
    LandingPaperPreviewState,
    LandingRepoPreviewState,
    build_landing_entry_header_html,
    build_landing_hero_html,
    build_landing_readiness_text,
    render_landing,
)
from labflow.ui.paper_preview import LandingPaperPreview, build_landing_paper_preview
from labflow.ui.pdf_viewer import render_pdf_viewer
from labflow.ui.quick_guide import (
    LandingQuickGuide,
    LandingQuickGuideState,
    build_landing_quick_guide,
    coerce_landing_quick_guide,
)
from labflow.ui.repo_preview import LandingRepoPreview, build_landing_repo_preview
from labflow.ui.sidebar import SidebarState, render_sidebar
from labflow.ui.styles import inject_styles

EVIDENCE_BUILDER = EvidenceBuilder()
REPORT_GENERATOR = ReportGenerator()
ALIGNMENT_CACHE_VERSION = "learning-output-v23"
__all__ = [
    "build_landing_paper_preview_state",
    "build_landing_quick_guide_state",
    "build_landing_repo_preview_state",
    "build_landing_entry_header_html",
    "build_landing_hero_html",
    "build_landing_readiness_text",
    "build_source_overview_html",
    "get_selected_section",
    "resolve_focus_section_index",
    "resolve_runtime_settings",
    "run",
    "should_render_source_grounding",
]


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
        page_title="LabFlow",
        page_icon="LF",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    inject_styles()

    current_route = st.session_state["current_route"]
    sidebar_state = get_sidebar_state(settings=settings, current_route=current_route)
    runtime_settings = resolve_runtime_settings(settings, sidebar_state)

    if current_route == "workspace":
        render_workspace(runtime_settings)
    elif current_route == "quick_guide":
        render_quick_guide(runtime_settings)
    else:
        render_landing(
            paper_preview_resolver=build_landing_paper_preview_state,
            repo_preview_resolver=build_landing_repo_preview_state,
        )


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
    st.session_state.setdefault("reading_note_history", {})
    st.session_state.setdefault("reading_note_markdown", "")


def build_landing_repo_preview_state(repo_path: str) -> LandingRepoPreviewState:
    """首页只在路径有效时展示轻量目录预览，避免输入过程中出现噪声。"""

    normalized_path = repo_path.strip()
    if not normalized_path:
        return LandingRepoPreviewState()

    path_obj = Path(normalized_path).expanduser()
    if not path_obj.exists():
        return LandingRepoPreviewState(hint="输入有效目录后，这里会显示项目结构预览。")
    if not path_obj.is_dir():
        return LandingRepoPreviewState(hint="当前路径不是目录，暂时无法预览代码结构。")

    try:
        preview = load_landing_repo_preview(str(path_obj))
    except (FileNotFoundError, ValueError):
        return LandingRepoPreviewState(hint="当前目录暂时无法解析，稍后可直接进入工作区再继续。")
    if preview is None:
        return LandingRepoPreviewState(hint="当前目录里还没有可预览的源码文件。")
    return LandingRepoPreviewState(preview=preview)


def build_landing_paper_preview_state(
    pdf_bytes: bytes | None,
    source_name: str | None,
) -> LandingPaperPreviewState:
    """首页论文信息卡优先依赖本地解析，外部元数据只做补充。"""

    if not pdf_bytes or not source_name:
        return LandingPaperPreviewState()

    try:
        preview = load_landing_paper_preview(pdf_bytes, source_name)
    except (RuntimeError, ValueError):
        return LandingPaperPreviewState(hint="论文已上传，进入工作区后仍可继续阅读。")
    if preview is None:
        return LandingPaperPreviewState(hint="已上传论文，暂时还没识别出稳定的标题信息。")
    return LandingPaperPreviewState(preview=preview)


def build_landing_quick_guide_state(
    pdf_bytes: bytes | None,
    source_name: str | None,
    settings: Settings,
) -> LandingQuickGuideState:
    """首页快速导读优先返回可读结论，没有论文时只提示下一步。"""

    if not pdf_bytes or not source_name:
        return LandingQuickGuideState(hint="先上传论文，再生成快速导读。")

    try:
        guide = load_landing_quick_guide(pdf_bytes, source_name, settings)
    except (RuntimeError, ValueError):
        return LandingQuickGuideState(hint="当前论文暂时无法生成导读，可先进入工作区继续阅读。")
    if guide is None:
        return LandingQuickGuideState(hint="当前论文还没有提取到足够稳定的导读信息。")
    return LandingQuickGuideState(guide=guide)


@st.cache_resource(show_spinner=False)
def get_llm_client(settings: Settings) -> LLMClient:
    return LLMClient(settings=settings)


@st.cache_resource(show_spinner=False)
def get_alignment_agent(settings: Settings) -> PlanAndExecuteAgent:
    return PlanAndExecuteAgent(llm_client=get_llm_client(settings))


def get_sidebar_state(settings: Settings, current_route: str) -> SidebarState:
    """首页不展示侧边栏，工作区再启用完整运行配置。"""

    if current_route == "workspace":
        sidebar_state = render_sidebar(settings)
        sync_sidebar_overrides(sidebar_state)
        return sidebar_state
    return SidebarState(
        uploaded_pdf_name=st.session_state.get("sidebar_uploaded_pdf_name"),
        uploaded_pdf_bytes=st.session_state.get("sidebar_uploaded_pdf_bytes"),
        git_repo_path=st.session_state.get("sidebar_git_repo_path", ""),
        api_key=st.session_state.get("sidebar_api_key") or None,
        base_url=st.session_state.get("sidebar_base_url", settings.base_url),
        model_name=st.session_state.get("sidebar_model_name", settings.model_name),
    )


def render_workspace(runtime_settings: Settings) -> None:
    workspace = get_workspace_state()
    render_workspace_header()

    if workspace.pdf_bytes is None:
        st.info("当前还没有可展示的工作区内容。先回到首页完成 PDF 和代码路径输入。")
        return

    left_column, right_column = st.columns([1.25, 1.15], gap="small")
    with left_column:
        selected_section = render_pdf_panel(workspace)
    with right_column:
        render_code_panel(workspace, selected_section, runtime_settings)


def render_quick_guide(runtime_settings: Settings) -> None:
    """渲染独立的论文导读页。"""

    pdf_bytes, pdf_name = resolve_pdf_source()
    preview = None
    guide = None
    if pdf_bytes and pdf_name:
        preview_state = build_landing_paper_preview_state(pdf_bytes, pdf_name)
        guide_state = build_landing_quick_guide_state(pdf_bytes, pdf_name, runtime_settings)
        preview = preview_state.preview
        guide = guide_state.guide

    render_quick_guide_page(
        preview=preview,
        guide=guide,
        source_name=pdf_name,
        has_repo_path=bool(resolve_git_repo_path()),
    )


def resolve_runtime_settings(base_settings: Settings, sidebar_state: SidebarState) -> Settings:
    """把侧边栏里的临时覆盖合并到当前运行时配置。"""

    return Settings(
        app_name=base_settings.app_name,
        app_env=base_settings.app_env,
        api_key=sidebar_state.api_key or base_settings.api_key,
        base_url=sidebar_state.base_url or base_settings.base_url,
        model_name=sidebar_state.model_name or base_settings.model_name,
    )


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
    if st.session_state.get("reading_note_history_signature") != signature:
        st.session_state["reading_note_history_signature"] = signature
        st.session_state["reading_note_history"] = {}
        st.session_state["reading_note_markdown"] = ""
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
def load_landing_repo_preview(repo_path: str) -> LandingRepoPreview | None:
    repo_result = load_repo_result(repo_path)
    return build_landing_repo_preview(
        relative_paths=tuple(source_file.relative_path for source_file in repo_result.source_files),
        source_type=repo_result.source_type,
        branch_name=repo_result.branch_name,
    )


@st.cache_data(show_spinner=False)
def load_landing_paper_preview(
    pdf_bytes: bytes,
    source_name: str,
) -> LandingPaperPreview | None:
    pdf_result = load_pdf_result(pdf_bytes, source_name)
    local_preview = build_landing_paper_preview(pdf_result=pdf_result, source_name=source_name)
    if local_preview is None:
        return None

    semantic_paper = SemanticScholarClient().search_by_title(local_preview.title)
    return build_landing_paper_preview(
        pdf_result=pdf_result,
        source_name=source_name,
        semantic_paper=semantic_paper,
    )


@st.cache_data(show_spinner=False)
def load_landing_quick_guide(
    pdf_bytes: bytes,
    source_name: str,
    settings: Settings,
) -> LandingQuickGuide | None:
    preview = load_landing_paper_preview(pdf_bytes, source_name)
    if preview is None:
        return None

    llm_client = None
    if settings.has_llm_credentials:
        try:
            llm_client = get_llm_client(settings)
        except RuntimeError:
            llm_client = None

    guide = build_landing_quick_guide(preview, llm_client=llm_client)
    return coerce_landing_quick_guide(guide, preview)


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


def resolve_focus_section_index(sections: tuple[PaperSection, ...], block_order: int) -> int | None:
    for index, section in enumerate(sections):
        if section.order == block_order:
            return index
        if section.block_orders and block_order in section.block_orders:
            return index
    return None


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
    section_index = resolve_focus_section_index(workspace.focus_sections, block_order)
    if section_index is not None:
        st.session_state["selected_section_index"] = section_index


def render_code_panel(
    workspace: WorkspaceState,
    selected_section: PaperSection | None,
    runtime_settings: Settings,
) -> None:
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
                runtime_settings=runtime_settings,
                event_handler=handle_agent_event,
            )
    except Exception as exc:  # noqa: BLE001
        trace_placeholder.empty()
        st.error(f"Agent 推理过程中出现异常：{exc}")
        st.info("本轮已停止推理。建议稍后重试，或先把注意力放回论文片段本身。")
        return

    trace_placeholder.empty()
    if alignment_result is None:
        st.warning("本轮模型没有稳定返回，我已停止展示中间推理链。")
        return
    record_reading_note_entry(
        st.session_state.get("workspace_signature", ""),
        selected_section,
        alignment_result,
    )
    render_code_canvas(alignment_result, trace_events, workspace, runtime_settings)


def get_semantic_alignment(
    *,
    workspace_signature: str,
    selected_section: PaperSection,
    repo_result: GitRepoParseResult,
    project_structure: str,
    runtime_settings: Settings,
    event_handler=None,
) -> AlignmentResult | None:
    cache_key = f"{ALIGNMENT_CACHE_VERSION}:{workspace_signature}:{selected_section.order}"
    cached_result = st.session_state["semantic_alignment_cache"].get(cache_key)
    if cached_result is not None:
        if event_handler is not None:
            event_handler({"kind": "cache_hit", "message": "命中缓存，直接复用上一次 Agent 结果。"})
        record_reading_note_entry(workspace_signature, selected_section, cached_result)
        return cached_result

    code_evidences = load_code_evidences(repo_result)
    alignment_result = get_alignment_agent(runtime_settings).run(
        selected_section,
        code_evidences,
        project_structure=project_structure,
        event_handler=event_handler,
    )
    st.session_state["semantic_alignment_cache"][cache_key] = alignment_result
    if alignment_result is not None:
        record_reading_note_entry(workspace_signature, selected_section, alignment_result)
    return alignment_result


def render_trace_panel(placeholder, trace_events: list[dict], *, finalized: bool) -> None:
    if not trace_events:
        placeholder.empty()
        return

    with placeholder.container():
        if finalized:
            with st.expander("查看推理链路", expanded=False):
                with st.container(height=300):
                    for event in trace_events:
                        render_trace_event(event)
            return

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


def render_code_canvas(
    alignment_result: AlignmentResult,
    trace_events: list[dict],
    workspace: WorkspaceState,
    runtime_settings: Settings,
) -> None:
    teach_tab, source_tab, trace_tab, note_tab = st.tabs(
        ["导师讲解", "源码导览", "推理链路", "阅读笔记"]
    )

    with teach_tab:
        st.markdown("### 【中文译文】")
        st.markdown(alignment_result.analysis)

        st.markdown("### 【核心要点】")
        st.markdown(alignment_result.semantic_evidence or "当前暂无重点提炼。")

        st.markdown("### 【术语百科】")
        st.markdown(alignment_result.research_supplement or "这一段没有特别需要额外展开的术语。")

    with source_tab:
        render_source_grounding_tab(alignment_result)

    with trace_tab:
        render_trace_tab(trace_events)

    with note_tab:
        render_reading_note_tab(workspace, runtime_settings)


def render_source_grounding_tab(alignment_result: AlignmentResult) -> None:
    if not should_render_source_grounding(alignment_result):
        st.info("当前还没有足够稳定的源码落地结果。")
        return

    source_guide = getattr(alignment_result, "source_guide", ())
    st.markdown("### 【源码导览】")
    st.caption("以下内容按当前论文片段聚合为更适合阅读的实现单元，不等同于唯一命中函数。")
    st.markdown(
        build_source_overview_html(alignment_result),
        unsafe_allow_html=True,
    )

    if source_guide:
        st.markdown("### 【相关实现】")
        for index, guide_item in enumerate(source_guide, start=1):
            title = (
                f"{index}. {guide_item.symbol_name} · "
                f"{guide_item.file_name} · L{guide_item.start_line}-L{guide_item.end_line}"
            )
            with st.expander(title, expanded=index == 1):
                st.markdown("**这段代码在做什么**")
                st.markdown(guide_item.summary)
                if guide_item.relevance_reason:
                    st.markdown("**和当前论文片段的关系**")
                    st.markdown(guide_item.relevance_reason)
                if guide_item.code_preview:
                    st.markdown("**代码入口**")
                    st.code(guide_item.code_preview, language=alignment_result.code_language)


def render_trace_tab(trace_events: list[dict]) -> None:
    if not trace_events:
        st.info("当前没有可展示的推理链路。")
        return

    with st.container(height=860):
        for event in trace_events:
            render_trace_event(event)


def render_reading_note_tab(workspace: WorkspaceState, runtime_settings: Settings) -> None:
    entries = get_reading_note_entries()
    if not entries:
        st.info("先在左侧点开几个论文片段，系统会把对应结果积累到这里。")
        return

    st.caption(f"当前已记录 {len(entries)} 个已读片段与对应代码。")
    if st.button("生成阅读笔记", use_container_width=True):
        st.session_state["reading_note_markdown"] = generate_reading_note_markdown(
            entries=entries,
            workspace=workspace,
            runtime_settings=runtime_settings,
        )

    markdown = st.session_state.get("reading_note_markdown", "").strip()
    if markdown:
        st.download_button(
            "下载 Markdown 笔记",
            data=markdown.encode("utf-8"),
            file_name="labflow-reading-notes.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.markdown(markdown)
    else:
        st.info("点击上方按钮生成可下载的阅读笔记。")


def should_render_source_grounding(alignment_result: AlignmentResult) -> bool:
    return (
        alignment_result.alignment_score >= 0.52
        and bool(alignment_result.implementation_chain.strip())
        and not alignment_result.code_snippet.startswith("# 当前未定位到对应源码")
    )


def record_reading_note_entry(
    workspace_signature: str,
    selected_section: PaperSection | None,
    alignment_result: AlignmentResult | None,
) -> None:
    if selected_section is None or alignment_result is None:
        return

    history: dict[str, ReadingNoteEntry] = st.session_state["reading_note_history"]
    history[f"{workspace_signature}:{selected_section.order}"] = ReadingNoteEntry(
        paper_section_title=selected_section.title,
        paper_section_content=selected_section.content,
        paper_section_page_number=selected_section.page_number,
        paper_section_order=selected_section.order,
        alignment_result=alignment_result,
    )
    st.session_state["reading_note_markdown"] = ""


def get_reading_note_entries() -> tuple[ReadingNoteEntry, ...]:
    history: dict[str, ReadingNoteEntry] = st.session_state.get("reading_note_history", {})
    return tuple(
        sorted(
            history.values(),
            key=lambda item: (item.paper_section_order, item.paper_section_page_number),
        )
    )


def build_reading_note_project_overview(
    workspace: WorkspaceState,
    entries: tuple[ReadingNoteEntry, ...],
) -> tuple[str, ...]:
    overview = [
        f"当前工作区已累计 {len(entries)} 个已读片段与对应代码。",
        "笔记内容来自当前会话里已经查过的论文片段和源码结果。",
    ]
    if workspace.project_structure:
        overview.append("当前项目结构索引已建立，适合继续追踪源码导览中的相关实现。")
    return tuple(overview)


def generate_reading_note_markdown(
    *,
    entries: tuple[ReadingNoteEntry, ...],
    workspace: WorkspaceState,
    runtime_settings: Settings,
) -> str:
    if not entries:
        return "# LabFlow 文献阅读笔记\n\n当前还没有已记录的片段。\n"

    llm_client = None
    try:
        llm_client = get_llm_client(runtime_settings)
    except Exception:  # noqa: BLE001
        llm_client = None

    return REPORT_GENERATOR.generate_literature_notes_markdown(
        entries=entries,
        llm_client=llm_client,
        project_overview=build_reading_note_project_overview(workspace, entries),
    )


def build_source_overview_html(alignment_result: AlignmentResult) -> str:
    source_guide = getattr(alignment_result, "source_guide", ())
    guide_cards = "".join(
        "".join(
            (
                '<div class="source-guide-card">',
                '<div class="source-guide-meta">',
                f'<span class="source-guide-symbol">{escape(item.symbol_name)}</span>',
                f'<span class="source-guide-range">{escape(item.file_name)} · '
                f"L{item.start_line}-L{item.end_line}</span>",
                "</div>",
                f'<div class="source-guide-summary">{escape(item.summary)}</div>',
                "</div>",
            )
        )
        for item in source_guide
    )
    legacy_text = " ".join(
        text
        for text in (
            alignment_result.implementation_chain,
            alignment_result.operator_alignment,
            alignment_result.shape_alignment,
            alignment_result.confidence_note,
        )
        if text
    )
    return (
        '<div class="source-overview-shell">'
        + (
            '<div class="source-guide-shell">'
            '<div class="source-section-title">关联模块</div>'
            f"{guide_cards}"
            "</div>"
            if guide_cards
            else ""
        )
        + f"<!-- {escape(legacy_text)} -->"
        "</div>"
    )


def _split_chain_sentences(chain: str) -> tuple[str, ...]:
    normalized = chain.replace("`", " ").replace("\n", " ")
    return tuple(sentence.strip() for sentence in normalized.split("。") if sentence.strip())


def sync_sidebar_overrides(sidebar_state: SidebarState) -> None:
    if sidebar_state.uploaded_pdf_bytes:
        st.session_state["sidebar_uploaded_pdf_bytes"] = sidebar_state.uploaded_pdf_bytes
        st.session_state["sidebar_uploaded_pdf_name"] = sidebar_state.uploaded_pdf_name
    if sidebar_state.git_repo_path:
        st.session_state["sidebar_git_repo_path"] = sidebar_state.git_repo_path
