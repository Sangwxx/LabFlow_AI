"""对齐器测试。"""

from labflow.reasoning.aligner import PaperCodeAligner, align_inputs
from labflow.reasoning.models import CodeEvidence, PaperSection


class FakeLLMClient:
    """我用一个可控假客户端把结构化结果锁死。"""

    def generate_json(self, *, system_prompt: str, user_prompt: str, **_: object) -> dict:
        assert "missing_implementation" in system_prompt
        assert "formula_mismatch" in user_prompt
        return {
            "alignment_score": 0.22,
            "match_type": "formula_mismatch",
            "analysis": "论文给出 alpha=0.70，但代码里把 alpha 写成了 0.30。",
            "improvement_suggestion": "把权重参数改回论文值，并补充回归测试。",
        }


def test_aligner_returns_structured_alignment_result() -> None:
    """对齐器应该把模型 JSON 整理成稳定结构。"""

    paper_sections = (
        PaperSection(
            title="3.2 Loss Weights",
            content="We set alpha = 0.70 and beta = 0.30 in the final loss.",
            level=2,
            page_number=3,
            order=1,
        ),
    )
    code_evidences = (
        CodeEvidence(
            file_name="trainer.py",
            code_snippet="alpha = 0.30\nbeta = 0.70",
            related_git_diff="-alpha = 0.70\n+alpha = 0.30",
            symbols=("alpha", "beta"),
            commit_context=("fix: adjust weights",),
        ),
    )

    results = align_inputs(
        paper_sections=paper_sections,
        code_evidences=code_evidences,
        llm_client=FakeLLMClient(),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].match_type == "formula_mismatch"
    assert results[0].paper_section_title == "3.2 Loss Weights"
    assert results[0].code_file_name == "trainer.py"


def test_aligner_can_process_candidates_directly() -> None:
    """直接传候选对时也应该能跑通。"""

    paper_sections = (
        PaperSection(
            title="4 Implementation",
            content="The paper requires a consistency regularizer.",
            level=1,
            page_number=4,
            order=2,
        ),
    )
    code_evidences = (
        CodeEvidence(
            file_name="model.py",
            code_snippet="loss = cls_loss",
            related_git_diff="+loss = cls_loss",
            symbols=("loss", "cls_loss"),
            commit_context=("feat: simplify loss",),
        ),
    )

    results = PaperCodeAligner(llm_client=FakeLLMClient()).align_inputs(
        paper_sections=paper_sections,
        code_evidences=code_evidences,
        top_k=1,
    )

    assert results[0].analysis
