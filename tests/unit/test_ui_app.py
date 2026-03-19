"""工作区语义代码画布测试。"""

from labflow.reasoning.models import AlignmentResult
from labflow.ui.app import build_highlighted_code_html


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
