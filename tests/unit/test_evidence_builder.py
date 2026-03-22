"""证据构建器测试。"""

import shutil
from pathlib import Path

from labflow.parsers.git_repo_parser import CommitInfo, GitRepoParseResult, SourceFile
from labflow.parsers.pdf_parser import PDFBlock, PDFParseResult
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import CodeEvidence, PaperSection


class FakeBrokenSemanticLLMClient:
    """我专门模拟返回非 JSON 的坏响应。"""

    def generate_json(self, **_: object) -> float:
        return 0.5


class FakeSemanticSummaryLLMClient:
    """我给代码卡片补一层更贴近论文术语的语义摘要。"""

    def generate_json(self, *, user_prompt: str, **_: object) -> dict:
        if "topo.py" in user_prompt:
            return {
                "summary": "这段代码负责维护导航过程中的拓扑地图并生成全局规划特征。",
                "responsibilities": [
                    "维护 topological map",
                    "聚合全局图结构供导航策略使用",
                ],
                "defined_symbols": ["build_memory_bank"],
                "called_symbols": ["update_graph", "encode_global_map"],
                "anchor_terms": ["topological map", "global planning", "graph memory"],
            }
        return {
            "summary": "这段代码主要做日志输出和通用工具处理。",
            "responsibilities": ["记录日志"],
            "defined_symbols": ["write_log"],
            "called_symbols": ["print"],
            "anchor_terms": ["logging"],
        }


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
                bbox=(20.0, 100.0, 420.0, 126.0),
            ),
            PDFBlock(
                kind="paragraph",
                text="The decoder predicts action logits conditioned on graph context.",
                page_number=1,
                order=2,
                font_size=11,
                bbox=(20.0, 170.0, 430.0, 196.0),
            ),
        ),
    )

    focus_sections = EvidenceBuilder().build_focus_sections(pdf_result)

    assert len(focus_sections) == 2
    assert focus_sections[0].title == "2 Method"
    assert focus_sections[0].order == 1
    assert "recurrent memory" in focus_sections[0].content


def test_evidence_builder_merges_adjacent_blocks_into_natural_paragraph() -> None:
    """相邻正文块如果属于同一段，应合并后再交给阅读工作区。"""

    pdf_result = PDFParseResult(
        source_name="paper.pdf",
        page_count=1,
        blocks=(
            PDFBlock(kind="title", text="Abstract", page_number=1, order=0, font_size=18),
            PDFBlock(
                kind="paragraph",
                text="Following language instructions to navigate in unseen environments",
                page_number=1,
                order=1,
                font_size=11,
                bbox=(20.0, 100.0, 400.0, 128.0),
            ),
            PDFBlock(
                kind="paragraph",
                text="is a challenging problem for autonomous embodied agents.",
                page_number=1,
                order=2,
                font_size=11,
                bbox=(20.0, 129.0, 420.0, 156.0),
            ),
        ),
    )

    focus_sections = EvidenceBuilder().build_focus_sections(pdf_result)

    assert len(focus_sections) == 1
    assert focus_sections[0].content == (
        "Following language instructions to navigate in unseen environments "
        "is a challenging problem for autonomous embodied agents."
    )
    assert focus_sections[0].block_orders == (1, 2)


def test_evidence_builder_merges_cross_page_continuation() -> None:
    """段落跨页延续时，我要继续把它还原成同一段。"""

    pdf_result = PDFParseResult(
        source_name="paper.pdf",
        page_count=2,
        blocks=(
            PDFBlock(kind="title", text="3 Method", page_number=1, order=0, font_size=18),
            PDFBlock(
                kind="paragraph",
                text="We maintain a latent map representation that keeps growing",
                page_number=1,
                order=1,
                font_size=11,
                bbox=(40.0, 710.0, 285.0, 780.0),
                page_width=595.0,
                page_height=842.0,
            ),
            PDFBlock(
                kind="paragraph",
                text="as the agent explores unseen environments during rollout.",
                page_number=2,
                order=2,
                font_size=11,
                bbox=(40.0, 48.0, 288.0, 118.0),
                page_width=595.0,
                page_height=842.0,
            ),
        ),
    )

    focus_sections = EvidenceBuilder().build_focus_sections(pdf_result)

    assert len(focus_sections) == 1
    assert focus_sections[0].content == (
        "We maintain a latent map representation that keeps growing "
        "as the agent explores unseen environments during rollout."
    )
    assert focus_sections[0].block_orders == (1, 2)


def test_evidence_builder_merges_cross_column_continuation() -> None:
    """双栏论文里，同一段跨列延续时也应继续合并。"""

    pdf_result = PDFParseResult(
        source_name="paper.pdf",
        page_count=1,
        blocks=(
            PDFBlock(kind="title", text="Abstract", page_number=1, order=0, font_size=18),
            PDFBlock(
                kind="paragraph",
                text="The proposed dual-scale encoder improves exploration efficiency",
                page_number=1,
                order=1,
                font_size=11,
                bbox=(40.0, 520.0, 280.0, 690.0),
                page_width=595.0,
                page_height=842.0,
            ),
            PDFBlock(
                kind="paragraph",
                text="by switching between local grounding and global planning cues.",
                page_number=1,
                order=2,
                font_size=11,
                bbox=(334.0, 72.0, 572.0, 160.0),
                page_width=595.0,
                page_height=842.0,
            ),
        ),
    )

    focus_sections = EvidenceBuilder().build_focus_sections(pdf_result)

    assert len(focus_sections) == 1
    assert focus_sections[0].content == (
        "The proposed dual-scale encoder improves exploration efficiency "
        "by switching between local grounding and global planning cues."
    )
    assert focus_sections[0].block_orders == (1, 2)


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


def test_evidence_builder_retrieves_candidates_from_semantic_cards() -> None:
    """代码卡片里的语义摘要应该能把论文机制召回到真正相关的实现文件。"""

    builder = EvidenceBuilder()
    paper_section = PaperSection(
        title="3.2 Global Planning",
        content=(
            "We maintain a topological map for efficient global planning in unseen environments."
        ),
        level=2,
        page_number=3,
        order=5,
    )
    semantic_index = builder.build_semantic_index_from_evidences(
        (
            CodeEvidence(
                file_name="topo.py",
                code_snippet="def build_memory_bank(x):\n    return update_graph(x)\n",
                related_git_diff="",
                symbols=("build_memory_bank", "update_graph"),
                commit_context=(),
                start_line=10,
                end_line=11,
            ),
            CodeEvidence(
                file_name="logger.py",
                code_snippet="def write_log(message):\n    print(message)\n",
                related_git_diff="",
                symbols=("write_log",),
                commit_context=(),
                start_line=1,
                end_line=2,
            ),
        ),
        llm_client=FakeSemanticSummaryLLMClient(),
    )

    candidates = builder.retrieve_semantic_candidates(
        paper_section,
        semantic_index,
        top_k=2,
    )

    assert candidates
    assert candidates[0].code_evidence.file_name == "topo.py"


def test_evidence_builder_reads_ast_logic_block_with_docstring() -> None:
    """read_code_segment 现在应该返回完整逻辑块，而不是生硬的行切片。"""

    builder = EvidenceBuilder()
    evidences = (
        CodeEvidence(
            file_name="encoder.py",
            code_snippet=(
                "def build_encoder(x):\n"
                '    """Fuse visual and language features."""\n'
                "    hidden = self.fuse(x)\n"
                "    return self.norm(hidden)\n"
            ),
            related_git_diff="",
            symbols=("build_encoder", "fuse", "norm"),
            commit_context=(),
            start_line=10,
            end_line=13,
            symbol_name="build_encoder",
            block_type="function",
            docstring="Fuse visual and language features.",
        ),
    )

    summary, _ = builder.read_logic_block(
        evidences,
        path="encoder.py",
        line_start=11,
        line_end=12,
    )

    assert "逻辑块类型: function" in summary
    assert "Fuse visual and language features." in summary
    assert "hidden = self.fuse(x)" in summary


def test_evidence_builder_can_find_definition_across_files() -> None:
    """Jedi 应该能把函数调用追到跨文件定义源头。"""

    repo_root = Path(".tmp/test-jedi-repo")
    if repo_root.exists():
        shutil.rmtree(repo_root, ignore_errors=True)
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "encoder.py").write_text(
        "from projection import project_tokens\n\n"
        "def build_encoder(x):\n"
        "    return project_tokens(x)\n",
        encoding="utf-8",
    )
    (repo_root / "projection.py").write_text(
        "def project_tokens(x):\n"
        '    """Project tokens into a shared space."""\n'
        "    return linear(x)\n",
        encoding="utf-8",
    )

    repo_result = GitRepoParseResult(
        repo_path=str(repo_root),
        branch_name="UNVERSIONED",
        recent_commits=(),
        working_tree_diff="",
        source_files=(
            SourceFile(
                relative_path="encoder.py",
                content=(repo_root / "encoder.py").read_text(encoding="utf-8"),
            ),
            SourceFile(
                relative_path="projection.py",
                content=(repo_root / "projection.py").read_text(encoding="utf-8"),
            ),
        ),
        source_type="directory",
    )

    builder = EvidenceBuilder()
    evidences = builder.build_code_evidences(repo_result)
    paper_section = PaperSection(
        title="3.1 Shared Projection",
        content="The encoder projects tokens into a shared feature space.",
        level=2,
        page_number=3,
        order=2,
    )

    observation, candidates = builder.find_definition_candidate(
        paper_section,
        evidences,
        symbol="project_tokens",
        file_path="encoder.py",
        line=4,
        column=11,
    )

    assert candidates
    assert candidates[0].code_evidence.file_name == "projection.py"
    assert "project_tokens" in observation
