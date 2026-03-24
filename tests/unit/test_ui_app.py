"""工作区与首页 UI 相关测试。"""

import streamlit as st

import labflow.ui.app as app_module
from labflow.config.settings import Settings
from labflow.reasoning.models import AlignmentResult, PaperSection, SourceGuideItem
from labflow.ui.app import (
    build_landing_entry_header_html,
    build_landing_hero_html,
    build_landing_paper_preview_state,
    build_landing_quick_guide_state,
    build_landing_readiness_text,
    build_landing_repo_preview_state,
    build_reading_note_project_overview,
    build_source_overview_html,
    generate_reading_note_markdown,
    get_reading_note_entries,
    get_selected_section,
    record_reading_note_entry,
    resolve_focus_section_index,
    resolve_runtime_settings,
    should_render_source_grounding,
)
from labflow.ui.paper_preview import LandingPaperPreview
from labflow.ui.repo_preview import build_landing_repo_preview, build_repo_preview_html
from labflow.ui.sidebar import SidebarState, _build_api_key_status


def build_result(
    *,
    title: str,
    file_name: str,
    score: float,
    match_type: str,
    analysis: str,
    suggestion: str,
) -> AlignmentResult:
    """构造测试用的对齐结果。"""

    return AlignmentResult(
        paper_section_title=title,
        code_file_name=file_name,
        alignment_score=score,
        match_type=match_type,
        analysis=analysis,
        improvement_suggestion=suggestion,
        retrieval_score=score,
    )


def test_build_source_overview_html_only_keeps_related_modules() -> None:
    """源码导览顶部概览区删除后，只保留关联模块。"""

    alignment_result = AlignmentResult(
        paper_section_title="3.2 Graph Policy",
        code_file_name="map_nav_src/models/vilmodel.py",
        alignment_score=0.83,
        match_type="strong_match",
        analysis="译文",
        semantic_evidence="重点",
        research_supplement="术语",
        implementation_chain="当前展示的源码片段来自 map_nav_src/models/vilmodel.py。",
        operator_alignment="这段代码里有图注意力相关算子。",
        shape_alignment="这里能看到图结构偏置被加到 attention mask 里。",
        confidence_note="建议人工核对 global encoder 的上游调用。",
        improvement_suggestion="",
        retrieval_score=0.45,
        code_snippet="graph_sprels = self.sprel_linear(x)",
        code_start_line=365,
        code_end_line=392,
        project_structure_context=(
            "项目主干目录：map_nav_src（2） / pretrain_src（2）。",
            "map_nav_src/models：承载模型主干、编码层和核心机制实现，当前目录共 7 个文件。",
        ),
        source_guide=(
            SourceGuideItem(
                file_name="map_nav_src/models/vilmodel.py",
                symbol_name="GraphLXRTXLayer.forward",
                block_type="method",
                start_line=384,
                end_line=395,
                summary="这段 forward 负责把语言特征、视觉特征和图结构偏置拼到同一层里。",
                responsibilities=(
                    "挂在 `GraphLXRTXLayer` 这一层级下。",
                    "代码范围位于 L384-L395，适合继续沿调用链复核。",
                ),
                relevance_reason="它命中的是图结构或全局规划相关实现，不是单纯的通用注意力壳层。",
            ),
        ),
    )

    html = build_source_overview_html(alignment_result)

    assert "map_nav_src/models/vilmodel.py" in html
    assert "L384-L395" in html
    assert "源码片段来自" in html
    assert "图结构偏置" in html
    assert "关联模块" in html
    assert "GraphLXRTXLayer.forward" in html
    assert "命中文件" not in html
    assert "代码范围" not in html
    assert "对齐分" not in html
    assert "召回分" not in html
    assert "项目结构定位" not in html
    assert "这段代码在做什么" not in html
    assert "它和论文片段的关系" not in html


def test_build_source_overview_html_is_backward_compatible_with_old_cache_result() -> None:
    """旧缓存里的 AlignmentResult 缺少新字段时，源码导览也不应报错。"""

    alignment_result = AlignmentResult(
        paper_section_title="3.2 Graph Policy",
        code_file_name="map_nav_src/models/vilmodel.py",
        alignment_score=0.83,
        match_type="strong_match",
        analysis="译文",
        semantic_evidence="重点",
        research_supplement="术语",
        implementation_chain="当前展示的源码片段来自 map_nav_src/models/vilmodel.py。",
        operator_alignment="这段代码里有图注意力相关算子。",
        shape_alignment="这里能看到图结构偏置被加到 attention mask 里。",
        confidence_note="建议人工核对 global encoder 的上游调用。",
        improvement_suggestion="",
        retrieval_score=0.45,
        code_snippet="graph_sprels = self.sprel_linear(x)",
        code_start_line=365,
        code_end_line=392,
    )
    del alignment_result.__dict__["project_structure_context"]
    del alignment_result.__dict__["source_guide"]

    html = build_source_overview_html(alignment_result)

    assert "关联模块" not in html


def test_reading_note_history_accumulates_and_sorts_entries() -> None:
    """阅读笔记应累计已读片段，并按论文顺序输出。"""

    st.session_state.clear()
    st.session_state["reading_note_history"] = {}
    st.session_state["reading_note_markdown"] = "old-note"

    section_later = PaperSection(
        title="3.2 Global Action Planning",
        content="The coarse-scale encoder predicts actions.",
        level=1,
        page_number=4,
        order=21,
    )
    section_earlier = PaperSection(
        title="3.1 Topological Mapping",
        content="The environment graph is initially unknown.",
        level=1,
        page_number=3,
        order=12,
    )
    alignment_later = build_result(
        title="3.2 Global Action Planning",
        file_name="vilmodel.py",
        score=0.76,
        match_type="partial_match",
        analysis="这段代码负责把粗尺度地图输入转成动作打分。",
        suggestion="继续追踪全局编码器调用链。",
    )
    alignment_earlier = build_result(
        title="3.1 Topological Mapping",
        file_name="graph_utils.py",
        score=0.81,
        match_type="strong_match",
        analysis="这段代码负责维护拓扑图状态。",
        suggestion="继续追踪图更新与路径查询。",
    )

    record_reading_note_entry("workspace-b", section_later, alignment_later)
    record_reading_note_entry("workspace-a", section_earlier, alignment_earlier)

    entries = get_reading_note_entries()

    assert st.session_state["reading_note_markdown"] == ""
    assert [entry.paper_section_order for entry in entries] == [12, 21]
    assert entries[0].paper_section_title == "3.1 Topological Mapping"
    assert entries[1].paper_section_title == "3.2 Global Action Planning"


def test_generate_reading_note_markdown_uses_report_generator(monkeypatch) -> None:
    """生成阅读笔记时应复用报告生成器和 LLM 入口。"""

    class FakeNoteLLM:
        def __init__(self) -> None:
            self.calls = 0

        def generate_text(self, **_: object) -> str:
            self.calls += 1
            return "# LabFlow 文献阅读笔记\n\n- 论文与代码实现之间存在清晰对应。\n"

    fake_llm = FakeNoteLLM()
    monkeypatch.setattr(app_module, "get_llm_client", lambda _settings: fake_llm)

    workspace = app_module.WorkspaceState(
        pdf_bytes=None,
        pdf_name=None,
        pdf_result=None,
        pdf_error=None,
        repo_result=None,
        repo_error=None,
        focus_sections=(),
        project_structure="map_nav_src 与 pretrain_src 构成主要结构。",
    )
    entry = app_module.ReadingNoteEntry(
        paper_section_title="3.1 Topological Mapping",
        paper_section_content="The environment graph is initially unknown to the agent.",
        paper_section_page_number=3,
        paper_section_order=12,
        alignment_result=build_result(
            title="3.1 Topological Mapping",
            file_name="graph_utils.py",
            score=0.81,
            match_type="strong_match",
            analysis="这段代码负责维护拓扑图状态。",
            suggestion="继续追踪图更新与路径查询。",
        ),
    )

    markdown = generate_reading_note_markdown(
        entries=(entry,),
        workspace=workspace,
        runtime_settings=Settings(
            app_env="dev",
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model_name="test-model",
        ),
    )
    overview = build_reading_note_project_overview(workspace, (entry,))

    assert fake_llm.calls == 1
    assert "# LabFlow 文献阅读笔记" in markdown
    assert "论文与代码实现之间存在清晰对应。" in markdown
    assert "当前工作区已累计 1 个已读片段与对应代码。" in overview[0]


def test_build_landing_entry_header_html_marks_ready_status() -> None:
    """首页入口卡应保留步骤、标题和状态。"""

    html = build_landing_entry_header_html(
        step_label="步骤 1",
        title="上传论文 PDF",
        description="上传后会抽取可点击段落。",
        status_text="已就绪",
        status_tone="ready",
    )

    assert "步骤 1" in html
    assert "上传论文 PDF" in html
    assert "已就绪" in html
    assert "entry-card-state-ready" in html


def test_build_landing_readiness_text_summarizes_missing_steps() -> None:
    """首页提示语应直接说明当前还缺什么。"""

    assert build_landing_readiness_text(has_pdf=False, has_repo_path=False) == "先完成这两项输入。"
    assert build_landing_readiness_text(has_pdf=True, has_repo_path=False) == "还差代码目录。"
    assert build_landing_readiness_text(has_pdf=False, has_repo_path=True) == "还差论文 PDF。"
    assert (
        build_landing_readiness_text(has_pdf=True, has_repo_path=True) == "已准备好，可以开始阅读。"
    )


def test_build_landing_hero_html_reflects_progress_state() -> None:
    """首页顶部只保留品牌、标题、副标题和状态。"""

    html = build_landing_hero_html(has_pdf=True, has_repo_path=False)

    assert "一体化科研助手" in html
    assert "上传论文、连接代码，然后直接开始阅读与定位。" in html
    assert "还差代码目录。" in html


def test_build_landing_repo_preview_groups_root_directories() -> None:
    """首页目录预览应按根目录聚合，并只展示轻量子项。"""

    preview = build_landing_repo_preview(
        relative_paths=(
            "map_nav_src/models/vilmodel.py",
            "map_nav_src/models/graph_utils.py",
            "map_nav_src/reverie/agent_obj.py",
            "pretrain_src/model/pretrain_cmt.py",
            "run.py",
        ),
        source_type="git",
        branch_name="main",
    )

    assert preview is not None
    assert preview.source_label == "Git 仓库预览 · main"
    assert preview.groups[0].root_name == "map_nav_src"
    assert preview.groups[0].file_count == 3
    assert preview.groups[0].children == ("models/", "reverie/")
    assert any(group.root_name == "根目录" for group in preview.groups)


def test_build_repo_preview_html_contains_grouped_structure() -> None:
    """首页目录预览 HTML 应保留分组标题与子项标签。"""

    preview = build_landing_repo_preview(
        relative_paths=(
            "map_nav_src/models/vilmodel.py",
            "map_nav_src/reverie/agent_obj.py",
        ),
        source_type="directory",
        branch_name="UNVERSIONED",
    )

    assert preview is not None
    html = build_repo_preview_html(preview)

    assert "代码目录预览" in html
    assert "map_nav_src" in html
    assert "models/" in html
    assert "reverie/" in html
    assert "文件" in html


def test_build_landing_repo_preview_state_returns_hint_for_missing_path() -> None:
    """路径无效时首页只给简短提示，不应该直接抛错。"""

    state = build_landing_repo_preview_state(r"E:\definitely-missing-path-for-labflow")

    assert state.preview is None
    assert state.hint == "输入有效目录后，这里会显示项目结构预览。"


def test_build_landing_paper_preview_state_handles_parser_failure(monkeypatch) -> None:
    """首页论文信息卡失败时应降级为简短提示，不影响上传入口。"""

    def _raise_preview_error(_pdf_bytes: bytes, _source_name: str):
        raise ValueError("preview failed")

    monkeypatch.setattr(app_module, "load_landing_paper_preview", _raise_preview_error)

    state = build_landing_paper_preview_state(b"%PDF-1.4", "demo.pdf")

    assert state.preview is None
    assert state.hint == "论文已上传，进入工作区后仍可继续阅读。"


def test_build_landing_quick_guide_state_returns_generated_guide(monkeypatch) -> None:
    """首页快速导读在论文可解析时应返回导读卡片。"""

    preview = LandingPaperPreview(
        title="Think Global, Act Local",
        authors=("Author A",),
        abstract="A dual-scale graph transformer is introduced for navigation.",
        source_label="已识别论文 · Think.pdf",
        meta_items=("15 页",),
        external_url=None,
    )

    monkeypatch.setattr(
        app_module,
        "load_landing_quick_guide",
        lambda *_args: app_module.build_landing_quick_guide(preview),
    )

    state = build_landing_quick_guide_state(
        b"%PDF-1.4",
        "demo.pdf",
        Settings(app_env="dev"),
    )

    assert state.guide is not None
    assert "双尺度图 Transformer" in state.guide.headline


def test_get_selected_section_supports_initial_silent_state() -> None:
    """未选择论文片段时，焦点状态应保持为空。"""

    sections = (
        PaperSection(
            title="1 Introduction",
            content="We introduce the task setting.",
            level=1,
            page_number=1,
            order=1,
        ),
        PaperSection(
            title="3 Method",
            content="We use a graph encoder for navigation.",
            level=1,
            page_number=3,
            order=2,
        ),
    )

    assert get_selected_section(sections) is None


def test_resolve_focus_section_index_supports_merged_block_orders() -> None:
    """合并后的自然段应允许点击其中任一原始块。"""

    sections = (
        PaperSection(
            title="Abstract",
            content="Following language instructions to navigate in unseen environments.",
            level=1,
            page_number=1,
            order=1,
            block_orders=(1, 2, 3),
        ),
        PaperSection(
            title="3 Method",
            content="We build a dual-scale graph encoder.",
            level=1,
            page_number=3,
            order=4,
            block_orders=(4,),
        ),
    )

    assert resolve_focus_section_index(sections, 2) == 0
    assert resolve_focus_section_index(sections, 4) == 1
    assert resolve_focus_section_index(sections, 99) is None


def test_source_grounding_shows_for_viable_grounding_result() -> None:
    """只要拿到可读的源码解释，源码导览就应显示。"""

    strong_result = AlignmentResult(
        paper_section_title="3.1 多头注意力",
        code_file_name="attention.py",
        alignment_score=0.9,
        match_type="strong_match",
        analysis="译文",
        semantic_evidence="重点",
        research_supplement="术语",
        implementation_chain="这段代码把 q/k/v 投影后再做多头拆分。",
        improvement_suggestion="",
        retrieval_score=0.2,
        code_snippet="q = self.q_proj(x)",
        code_start_line=10,
        code_end_line=10,
    )
    partial_result = AlignmentResult(
        paper_section_title="3.2 Graph Policy",
        code_file_name="policy.py",
        alignment_score=0.61,
        match_type="partial_match",
        analysis="译文",
        semantic_evidence="重点",
        research_supplement="术语",
        implementation_chain="这段代码负责把局部观测和全局图信息拼起来后做导航决策。",
        improvement_suggestion="",
        retrieval_score=0.12,
        code_snippet="state = fuse(local_obs, global_map)\nreturn policy_head(state)\n",
        code_start_line=11,
        code_end_line=12,
    )
    weak_result = AlignmentResult(
        paper_section_title="1 Introduction",
        code_file_name="未定位到本地实现",
        alignment_score=0.35,
        match_type="missing_implementation",
        analysis="译文",
        semantic_evidence="重点",
        research_supplement="术语",
        implementation_chain="",
        improvement_suggestion="",
        retrieval_score=0.0,
        code_snippet="# 当前未定位到对应源码\n",
        code_start_line=1,
        code_end_line=1,
    )

    assert should_render_source_grounding(strong_result) is True
    assert should_render_source_grounding(partial_result) is True
    assert should_render_source_grounding(weak_result) is False


def test_resolve_runtime_settings_prefers_sidebar_overrides() -> None:
    """侧边栏里的覆盖配置应进入运行时设置。"""

    base_settings = Settings(
        app_env="dev",
        api_key="env-key",
        base_url="https://api.example.com/v1",
        model_name="base-model",
    )
    sidebar_state = SidebarState(
        uploaded_pdf_name=None,
        uploaded_pdf_bytes=None,
        git_repo_path=r"E:\VLN-DUET-main",
        api_key="override-key",
        base_url="https://override.example.com/v1",
        model_name="override-model",
    )

    runtime_settings = resolve_runtime_settings(base_settings, sidebar_state)

    assert runtime_settings.api_key == "override-key"
    assert runtime_settings.base_url == "https://override.example.com/v1"
    assert runtime_settings.model_name == "override-model"


def test_resolve_runtime_settings_falls_back_to_env_key() -> None:
    """未手动输入密钥时，继续使用环境变量配置。"""

    base_settings = Settings(
        app_env="dev",
        api_key="env-key",
        base_url="https://api.example.com/v1",
        model_name="base-model",
    )
    sidebar_state = SidebarState(
        uploaded_pdf_name=None,
        uploaded_pdf_bytes=None,
        git_repo_path="",
        api_key=None,
        base_url="https://api.example.com/v1",
        model_name="base-model",
    )

    runtime_settings = resolve_runtime_settings(base_settings, sidebar_state)

    assert runtime_settings.api_key == "env-key"


def test_build_api_key_status_does_not_require_echoing_secret() -> None:
    """状态提示只描述来源，不应回显真实密钥。"""

    assert (
        _build_api_key_status(has_env_api_key=True, has_session_override=False)
        == "已检测到 `.env` 中的 API Key，当前输入框不会回显真实值。"
    )
    assert (
        _build_api_key_status(has_env_api_key=True, has_session_override=True)
        == "当前会话已应用手动输入的 API Key，页面不会回显具体值。"
    )
    assert _build_api_key_status(has_env_api_key=False, has_session_override=False) is None
