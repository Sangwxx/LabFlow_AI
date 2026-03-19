"""证据构建器测试。"""

from labflow.parsers.git_repo_parser import CommitInfo, GitRepoParseResult
from labflow.parsers.pdf_parser import PDFBlock, PDFParseResult
from labflow.reasoning.evidence_builder import EvidenceBuilder


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
