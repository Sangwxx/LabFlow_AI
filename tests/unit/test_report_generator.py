"""报告生成器测试。"""

from labflow.reasoning.models import AlignmentResult
from labflow.reporting import ReadingNoteEntry, ReportGenerator


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


class FakeLiteratureNoteLLMClient:
    """返回一段简洁的阅读笔记 Markdown。"""

    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, **_: object) -> str:
        self.calls += 1
        return (
            "# LabFlow 文献阅读笔记\n\n"
            "## 综述\n"
            "- 论文片段和代码实现之间存在明确对应。\n\n"
            "## 片段说明\n"
            "- 论文说了什么：拓扑地图不断累积。\n"
            "- 代码做了什么：更新图结构并计算位置特征。\n"
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


def test_report_generator_can_build_literature_notes_markdown() -> None:
    """阅读笔记应支持把论文片段和代码一起整理成可下载 Markdown。"""

    entry = ReadingNoteEntry(
        paper_section_title="3.1 Topological Mapping",
        paper_section_content="The environment graph is initially unknown to the agent.",
        paper_section_page_number=3,
        paper_section_order=12,
        alignment_result=build_result(
            title="3.1 Topological Mapping",
            file_name="graph_utils.py",
            score=0.81,
            match_type="strong_match",
            analysis="这段代码更新拓扑地图并维护节点状态。",
            suggestion="保持当前图更新逻辑。",
        ),
    )

    markdown = ReportGenerator().generate_literature_notes_markdown(
        entries=(entry,),
        llm_client=FakeLiteratureNoteLLMClient(),
        project_overview=("当前工作区已记录 1 个片段与对应代码。",),
    )

    assert "# LabFlow 文献阅读笔记" in markdown
    assert "论文片段和代码实现之间存在明确对应" in markdown


def test_report_generator_falls_back_when_note_llm_is_missing() -> None:
    """没有 LLM 时，报告仍应能回退到稳定的 Markdown 结构。"""

    entry = ReadingNoteEntry(
        paper_section_title="3.2 Global Action Planning",
        paper_section_content="The coarse-scale encoder predicts actions.",
        paper_section_page_number=4,
        paper_section_order=21,
        alignment_result=build_result(
            title="3.2 Global Action Planning",
            file_name="vilmodel.py",
            score=0.76,
            match_type="partial_match",
            analysis="这段代码负责全局地图节点输入和动作预测。",
            suggestion="继续追踪全局编码器的调用链。",
        ),
    )

    markdown = ReportGenerator().generate_literature_notes_markdown(
        entries=(entry,),
        llm_client=None,
        project_overview=(),
    )

    assert "# LabFlow 文献阅读笔记" in markdown
    assert "论文要点" in markdown
    assert "后续阅读建议" in markdown


def test_report_generator_keeps_llm_path_for_large_note_batch() -> None:
    """条目较多时仍应保留完整 LLM 笔记生成路径。"""

    llm_client = FakeLiteratureNoteLLMClient()
    entries = tuple(
        ReadingNoteEntry(
            paper_section_title=f"片段 {index}",
            paper_section_content="The model builds a graph-aware navigation representation.",
            paper_section_page_number=index,
            paper_section_order=index,
            alignment_result=build_result(
                title=f"片段 {index}",
                file_name="graph_utils.py",
                score=0.75,
                match_type="partial_match",
                analysis="这段代码负责更新图结构并维护当前状态。",
                suggestion="继续补充上下文，确认该实现是否覆盖完整机制。",
            ),
        )
        for index in range(1, 6)
    )

    markdown = ReportGenerator().generate_literature_notes_markdown(
        entries=entries,
        llm_client=llm_client,
        project_overview=(),
    )

    assert llm_client.calls == 1
    assert "# LabFlow 文献阅读笔记" in markdown
