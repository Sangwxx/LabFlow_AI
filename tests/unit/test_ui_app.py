"""工作区语义代码画布测试。"""

from labflow.config.settings import Settings
from labflow.reasoning.models import AlignmentResult, PaperSection
from labflow.ui.app import (
    build_landing_entry_header_html,
    build_landing_readiness_text,
    build_highlighted_code_html,
    build_landing_hero_html,
    build_source_overview_html,
    get_selected_section,
    resolve_focus_section_index,
    resolve_runtime_settings,
    should_render_source_grounding,
)
from labflow.ui.sidebar import SidebarState, _build_api_key_status


def test_build_highlighted_code_html_marks_semantic_lines() -> None:
    """语义命中的逻辑行应该在代码画布里被显式高亮。"""

    alignment_result = AlignmentResult(
        paper_section_title="3.1 多头注意力",
        code_file_name="attention.py",
        alignment_score=0.91,
        match_type="strong_match",
        analysis="这段实现完成了论文中的多头注意力。",
        semantic_evidence="q/k/v 投影、按 8 个 head 重排、拼接后输出映射，逻辑完整闭环。",
        improvement_suggestion="可补一段注释说明 head 的配置来源。",
        retrieval_score=0.18,
        highlighted_line_numbers=(24, 26),
        code_snippet=(
            "q = self.q_proj(x)\n"
            "k = self.k_proj(x)\n"
            "heads = rearrange(q, 'b n (h d) -> b h n d', h=8)"
        ),
        code_start_line=24,
        code_end_line=26,
    )

    html = build_highlighted_code_html(alignment_result)

    assert "attention.py" in html
    assert "code-line-highlight" in html
    assert ">24<" in html
    assert ">26<" in html
    assert "rearrange" in html
    assert html.startswith('<div class="semantic-code-shell">')


def test_build_source_overview_html_contains_grounding_summary() -> None:
    """源码定位概览卡应把文件、范围和说明收拢到同一个区域。"""

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
        shape_alignment="这里能看到图结构偏置被加到 attention mask 上。",
        confidence_note="建议人工核对 global encoder 的上游调用。",
        improvement_suggestion="",
        retrieval_score=0.45,
        code_snippet="graph_sprels = self.sprel_linear(x)",
        code_start_line=365,
        code_end_line=392,
    )

    html = build_source_overview_html(alignment_result)

    assert "map_nav_src/models/vilmodel.py" in html
    assert "L365-L392" in html
    assert "源码片段来自" in html
    assert "图结构偏置" in html


def test_build_landing_entry_header_html_marks_ready_status() -> None:
    """首页入口卡应该保留步骤、标题和精简状态。"""

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
    """首页提示语应直接说清当前还缺什么。"""

    assert build_landing_readiness_text(has_pdf=False, has_repo_path=False) == "先完成这两项输入。"
    assert build_landing_readiness_text(has_pdf=True, has_repo_path=False) == "还差代码目录。"
    assert build_landing_readiness_text(has_pdf=False, has_repo_path=True) == "还差论文 PDF。"
    assert build_landing_readiness_text(has_pdf=True, has_repo_path=True) == "已准备好，可以开始阅读。"


def test_build_landing_hero_html_reflects_progress_state() -> None:
    """首页顶部应保持克制，只展示品牌、标题、副标题和状态。"""

    html = build_landing_hero_html(has_pdf=True, has_repo_path=False)

    assert "论文与代码，对齐阅读" in html
    assert "准备好 PDF 和代码目录后，直接进入工作区。" in html
    assert "还差代码目录。" in html


def test_get_selected_section_supports_initial_silent_state() -> None:
    """未选择论文片段时，左侧焦点栏应保持静默。"""

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
    """合并后的自然段应该允许我点击其中任意一个原始块。"""

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
    """只要拿到了可信源码解释，源码落地模块就应该展示。"""

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
    """侧边栏里手动覆盖的模型配置应该真正进入运行时。"""

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
    """如果侧边栏没有手动输入密钥，就继续使用环境变量配置。"""

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
    """状态提示只描述来源，不应该要求把真实密钥显示回页面。"""

    assert (
        _build_api_key_status(has_env_api_key=True, has_session_override=False)
        == "已检测到 `.env` 中的 API Key，当前输入框不会回显真实值。"
    )
    assert (
        _build_api_key_status(has_env_api_key=True, has_session_override=True)
        == "当前会话已应用手动输入的 API Key，页面不会回显具体值。"
    )
    assert _build_api_key_status(has_env_api_key=False, has_session_override=False) is None
