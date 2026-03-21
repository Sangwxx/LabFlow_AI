"""工作区语义代码画布测试。"""

from labflow.reasoning.models import AlignmentResult, PaperSection
from labflow.ui.app import (
    build_highlighted_code_html,
    get_selected_section,
    resolve_focus_section_index,
    should_render_source_grounding,
)


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


def test_source_grounding_only_shows_for_strong_alignment() -> None:
    """源码落地模块只应在强关联命中时展示。"""

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
    assert should_render_source_grounding(weak_result) is False
