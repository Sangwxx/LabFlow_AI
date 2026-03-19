"""报告生成器测试。"""

from labflow.reasoning.models import AlignmentResult
from labflow.reporting import ReportGenerator


def build_result(
    *,
    title: str,
    file_name: str,
    score: float,
    match_type: str,
    analysis: str,
    suggestion: str,
) -> AlignmentResult:
    """构造测试用结果。"""

    return AlignmentResult(
        paper_section_title=title,
        code_file_name=file_name,
        alignment_score=score,
        match_type=match_type,
        analysis=analysis,
        improvement_suggestion=suggestion,
        retrieval_score=score,
    )


def test_report_generator_outputs_markdown_sections() -> None:
    """周报里应该有项目概况、高风险和建议三个核心区块。"""

    results = (
        build_result(
            title="3.2 损失函数权重",
            file_name="trainer.py",
            score=0.2,
            match_type="formula_mismatch",
            analysis="论文和代码的 alpha、beta 系数不一致。",
            suggestion="修正权重参数并补测试。",
        ),
        build_result(
            title="4 Implementation",
            file_name="model.py",
            score=0.92,
            match_type="strong_match",
            analysis="实现与论文描述一致。",
            suggestion="保持现有实现并持续回归验证。",
        ),
    )

    markdown = ReportGenerator().generate_markdown(
        results=results,
        project_overview=("输入为文本型 PDF + 本地 Git 仓库。",),
    )

    assert "# LabFlow AI 审计周报" in markdown
    assert "## 项目概况" in markdown
    assert "## 🔴 高风险错配项" in markdown
    assert "## 🟢 一致性良好项" in markdown
    assert "## 改进建议" in markdown
    assert "trainer.py" in markdown
    assert "model.py" in markdown


def test_report_generator_summary_uses_ten_point_scale() -> None:
    """总体置信度应该按 10 分制汇总。"""

    results = (
        build_result(
            title="Method",
            file_name="a.py",
            score=0.5,
            match_type="partial_match",
            analysis="需要补证据。",
            suggestion="补更多上下文。",
        ),
        build_result(
            title="Results",
            file_name="b.py",
            score=0.7,
            match_type="partial_match",
            analysis="部分一致。",
            suggestion="继续校准参数。",
        ),
    )

    summary = ReportGenerator().build_summary(results)

    assert summary.total_items == 2
    assert summary.high_risk_items == 1
    assert summary.overall_confidence == 6.0
