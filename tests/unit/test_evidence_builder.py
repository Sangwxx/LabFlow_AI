"""证据构建器测试。"""

from labflow.parsers.git_repo_parser import CommitInfo, GitRepoParseResult, SourceFile
from labflow.parsers.pdf_parser import PDFBlock, PDFParseResult
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import CodeEvidence


class FakeBrokenSemanticLLMClient:
    """我专门模拟返回非 JSON 的坏响应。"""

    def generate_json(self, **_: object) -> float:
        return 0.5


def test_evidence_builder_builds_sections_from_pdf_blocks() -> None:
    """我会把标题块和正文块重新拼成章节。"""

    pdf_result = PDFParseResult(
        source_name="paper.pdf",
        page_count=2,
        blocks=(
            PDFBlock(kind="title", text="1 Method", page_number=1, order=0, font_size=18),
            PDFBlock(
                kind="paragraph",
                text="The method uses alpha and beta to balance losses.",
                page_number=1,
                order=1,
                font_size=11,
            ),
            PDFBlock(kind="title", text="2 Experiments", page_number=2, order=2, font_size=17),
            PDFBlock(
                kind="paragraph",
                text="We compare against strong baselines.",
                page_number=2,
                order=3,
                font_size=11,
            ),
        ),
    )

    sections = EvidenceBuilder().build_paper_sections(pdf_result)

    assert len(sections) == 2
    assert sections[0].title == "1 Method"
    assert "alpha and beta" in sections[0].content
    assert sections[0].level == 1


def test_evidence_builder_builds_focus_sections_for_paragraph_clicks() -> None:
    """页图点击需要段落级焦点对象，我会把每个正文块挂上最近标题。"""

    pdf_result = PDFParseResult(
        source_name="paper.pdf",
        page_count=1,
        blocks=(
            PDFBlock(kind="title", text="2 Method", page_number=1, order=0, font_size=18),
            PDFBlock(
                kind="paragraph",
                text="The encoder uses recurrent memory for long horizon planning.",
                page_number=1,
                order=1,
                font_size=11,
            ),
            PDFBlock(
                kind="paragraph",
                text="The decoder predicts action logits conditioned on graph context.",
                page_number=1,
                order=2,
                font_size=11,
            ),
        ),
    )

    focus_sections = EvidenceBuilder().build_focus_sections(pdf_result)

    assert len(focus_sections) == 2
    assert focus_sections[0].title == "2 Method"
    assert focus_sections[0].order == 1
    assert "recurrent memory" in focus_sections[0].content


def test_evidence_builder_recalls_relevant_section_from_diff() -> None:
    """代码里的标识符应该能把相关论文章节顶出来。"""

    pdf_result = PDFParseResult(
        source_name="paper.pdf",
        page_count=1,
        blocks=(
            PDFBlock(kind="title", text="3.2 Loss Weights", page_number=1, order=0, font_size=18),
            PDFBlock(
                kind="paragraph",
                text="We set alpha = 0.70 and beta = 0.30 in the final loss function.",
                page_number=1,
                order=1,
                font_size=11,
            ),
            PDFBlock(kind="title", text="4 Results", page_number=1, order=2, font_size=18),
            PDFBlock(
                kind="paragraph",
                text="The model outperforms baselines on standard metrics.",
                page_number=1,
                order=3,
                font_size=11,
            ),
        ),
    )
    repo_result = GitRepoParseResult(
        repo_path="D:/repo",
        branch_name="main",
        recent_commits=(
            CommitInfo(
                hexsha="a" * 40,
                short_sha="aaaaaaa",
                author_name="LabFlow Dev",
                authored_at="2026-03-19T10:00:00",
                summary="feat: tune alpha beta weights",
            ),
        ),
        working_tree_diff=(
            "diff --git a/trainer.py b/trainer.py\n"
            "index 123..456 100644\n"
            "--- a/trainer.py\n"
            "+++ b/trainer.py\n"
            "@@ -10,2 +10,2 @@\n"
            "-alpha = 0.70\n"
            "+alpha = 0.30\n"
            "-beta = 0.30\n"
            "+beta = 0.70\n"
        ),
    )

    candidates = EvidenceBuilder().build_alignment_candidates(pdf_result, repo_result, top_k=1)

    assert len(candidates) == 1
    assert candidates[0].paper_section.title == "3.2 Loss Weights"
    assert candidates[0].code_evidence.file_name == "trainer.py"
    assert candidates[0].retrieval_score > 0


def test_evidence_builder_builds_code_evidence_from_source_files() -> None:
    """右栏联动时，我要拿到真实代码片段和行号，而不是只有 diff。"""

    repo_result = GitRepoParseResult(
        repo_path="D:/repo",
        branch_name="UNVERSIONED",
        recent_commits=(),
        working_tree_diff="",
        source_files=(
            SourceFile(
                relative_path="trainer.py",
                content=(
                    "import math\n\n"
                    "def build_loss(alpha: float, beta: float):\n"
                    "    cls_loss = 0.1\n"
                    "    reg_loss = 0.2\n"
                    "    return alpha * cls_loss + beta * reg_loss\n"
                ),
            ),
        ),
        source_type="directory",
    )

    evidences = EvidenceBuilder().build_code_evidences(repo_result)

    assert evidences
    assert evidences[0].file_name == "trainer.py"
    assert evidences[0].start_line == 1
    assert evidences[0].end_line >= evidences[0].start_line


def test_evidence_builder_builds_semantic_index_from_evidences() -> None:
    """语义索引至少要把职责、定义符号和调用符号整理出来。"""

    semantic_index = EvidenceBuilder().build_semantic_index_from_evidences(
        (
            CodeEvidence(
                file_name="attention.py",
                code_snippet=(
                    "def build_attention(x):\n    q = self.q_proj(x)\n    return self.out_proj(q)\n"
                ),
                related_git_diff="",
                symbols=("build_attention", "q_proj", "out_proj"),
                commit_context=("feat: add attention",),
                start_line=24,
                end_line=26,
            ),
        )
    )

    assert len(semantic_index) == 1
    assert semantic_index[0].summary
    assert "build_attention" in semantic_index[0].defined_symbols
    assert "q_proj" in semantic_index[0].called_symbols


def test_evidence_builder_handles_non_dict_semantic_payload() -> None:
    """语义摘要返回坏响应时，构建器也要稳稳退回兜底逻辑。"""

    semantic_index = EvidenceBuilder().build_semantic_index_from_evidences(
        (
            CodeEvidence(
                file_name="attention.py",
                code_snippet=(
                    "def build_attention(x):\n    q = self.q_proj(x)\n    return self.out_proj(q)\n"
                ),
                related_git_diff="",
                symbols=("build_attention", "q_proj", "out_proj"),
                commit_context=("feat: add attention",),
                start_line=24,
                end_line=26,
            ),
        ),
        llm_client=FakeBrokenSemanticLLMClient(),
    )

    assert len(semantic_index) == 1
    assert semantic_index[0].summary
    assert "build_attention" in semantic_index[0].defined_symbols


def test_evidence_builder_can_trace_related_candidates_by_symbol_chain() -> None:
    """当 Agent 要追函数调用链时，证据构建器应能沿符号找到上游定义。"""

    builder = EvidenceBuilder()
    paper_section = builder.build_focus_sections(
        PDFParseResult(
            source_name="paper.pdf",
            page_count=1,
            blocks=(
                PDFBlock(kind="title", text="3 Method", page_number=1, order=0, font_size=18),
                PDFBlock(
                    kind="paragraph",
                    text=(
                        "The model projects q and uses output projection "
                        "to close the attention block."
                    ),
                    page_number=1,
                    order=1,
                    font_size=11,
                ),
            ),
        )
    )[0]
    semantic_index = builder.build_semantic_index_from_evidences(
        (
            CodeEvidence(
                file_name="attention.py",
                code_snippet="def build_attention(x):\n    q = q_proj(x)\n    return out_proj(q)\n",
                related_git_diff="",
                symbols=("build_attention", "q_proj", "out_proj"),
                commit_context=(),
                start_line=24,
                end_line=26,
            ),
            CodeEvidence(
                file_name="projection.py",
                code_snippet="def q_proj(x):\n    return linear(x)\n",
                related_git_diff="",
                symbols=("q_proj", "linear"),
                commit_context=(),
                start_line=6,
                end_line=7,
            ),
        )
    )

    traced_candidates = builder.trace_related_candidates(
        paper_section,
        semantic_index,
        trace_symbols=("q_proj",),
        seen_candidate_ids={"attention.py:24-26"},
        limit=3,
    )

    assert traced_candidates
    assert traced_candidates[0].code_evidence.file_name == "projection.py"
