"""基于轻量检索的证据构建器。"""

from __future__ import annotations

import math
import re
from collections import Counter

from labflow.parsers.git_repo_parser import GitRepoParseResult
from labflow.parsers.pdf_parser import PDFBlock, PDFParseResult
from labflow.reasoning.models import AlignmentCandidate, CodeEvidence, PaperSection

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")
DIFF_HEADER_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+)$")
HEADING_NUMBER_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)")


class EvidenceBuilder:
    """我先用 BM25 风格召回把最可能相关的证据对压出来，再交给模型做精判。"""

    def build_paper_sections(self, pdf_result: PDFParseResult) -> tuple[PaperSection, ...]:
        """把 PDF 结构块重组为章节对象。"""

        sections: list[PaperSection] = []
        current_title = "未命名章节"
        current_level = 1
        current_page = 1
        current_order = 0
        paragraph_buffer: list[str] = []

        for block in pdf_result.blocks:
            if block.kind == "title":
                self._flush_section(
                    sections=sections,
                    title=current_title,
                    level=current_level,
                    page_number=current_page,
                    order=current_order,
                    paragraph_buffer=paragraph_buffer,
                )
                current_title = block.text
                current_level = self._infer_section_level(block)
                current_page = block.page_number
                current_order = block.order
                paragraph_buffer = []
                continue

            paragraph_buffer.append(block.text)

        self._flush_section(
            sections=sections,
            title=current_title,
            level=current_level,
            page_number=current_page,
            order=current_order,
            paragraph_buffer=paragraph_buffer,
        )
        return tuple(section for section in sections if section.content.strip())

    def build_code_evidences(self, repo_result: GitRepoParseResult) -> tuple[CodeEvidence, ...]:
        """把 Git diff 和提交信息整理成代码侧证据。"""

        commit_context = tuple(commit.summary for commit in repo_result.recent_commits)
        evidences = self._parse_diff_evidences(repo_result.working_tree_diff, commit_context)
        if evidences:
            return tuple(evidences)

        fallback_text = "\n".join(commit_context)
        fallback_symbols = tuple(self._tokenize_text(fallback_text))
        if not fallback_text:
            fallback_text = "当前工作区没有 diff，最近提交也为空。"
        return (
            CodeEvidence(
                file_name="working-tree-clean",
                code_snippet=fallback_text,
                related_git_diff=repo_result.working_tree_diff,
                symbols=fallback_symbols,
                commit_context=commit_context,
            ),
        )

    def build_alignment_candidates(
        self,
        pdf_result: PDFParseResult,
        repo_result: GitRepoParseResult,
        top_k: int = 2,
    ) -> tuple[AlignmentCandidate, ...]:
        """用轻量 BM25 检索把高相关候选对召回出来。"""

        paper_sections = self.build_paper_sections(pdf_result)
        code_evidences = self.build_code_evidences(repo_result)
        return self.build_alignment_candidates_from_inputs(
            paper_sections, code_evidences, top_k=top_k
        )

    def build_alignment_candidates_from_inputs(
        self,
        paper_sections: tuple[PaperSection, ...],
        code_evidences: tuple[CodeEvidence, ...],
        top_k: int = 2,
    ) -> tuple[AlignmentCandidate, ...]:
        """对任意章节/证据输入执行召回。"""

        if not paper_sections or not code_evidences:
            return ()

        document_tokens = [self._tokenize_text(section.combined_text) for section in paper_sections]
        avg_doc_len = sum(len(tokens) for tokens in document_tokens) / len(document_tokens)
        doc_frequencies = self._build_document_frequencies(document_tokens)

        candidates: list[AlignmentCandidate] = []
        for evidence in code_evidences:
            query_tokens = list(
                dict.fromkeys(
                    evidence.symbols or tuple(self._tokenize_text(evidence.combined_text))
                )
            )
            scored_sections: list[tuple[PaperSection, float]] = []
            for section, tokens in zip(paper_sections, document_tokens, strict=False):
                score = self._bm25_score(
                    query_tokens=query_tokens,
                    document_tokens=tokens,
                    avg_doc_len=avg_doc_len,
                    document_count=len(document_tokens),
                    doc_frequencies=doc_frequencies,
                )
                scored_sections.append((section, score))

            scored_sections.sort(key=lambda item: item[1], reverse=True)
            for section, score in scored_sections[:top_k]:
                candidates.append(
                    AlignmentCandidate(
                        paper_section=section,
                        code_evidence=evidence,
                        retrieval_score=round(score, 4),
                    )
                )

        candidates.sort(key=lambda item: item.retrieval_score, reverse=True)
        return tuple(candidates)

    def _parse_diff_evidences(
        self,
        diff_text: str,
        commit_context: tuple[str, ...],
    ) -> list[CodeEvidence]:
        """按文件粒度拆分 diff。"""

        if not diff_text.strip():
            return []

        evidences: list[CodeEvidence] = []
        current_file: str | None = None
        current_diff_lines: list[str] = []
        current_code_lines: list[str] = []

        def flush_current() -> None:
            nonlocal current_file, current_diff_lines, current_code_lines
            if not current_file or not current_diff_lines:
                current_file = None
                current_diff_lines = []
                current_code_lines = []
                return

            code_snippet = "\n".join(current_code_lines).strip()
            related_git_diff = "\n".join(current_diff_lines).strip()
            symbols = tuple(
                self._extract_symbols(f"{current_file}\n{code_snippet}\n{related_git_diff}")
            )
            evidences.append(
                CodeEvidence(
                    file_name=current_file,
                    code_snippet=code_snippet or related_git_diff,
                    related_git_diff=related_git_diff,
                    symbols=symbols,
                    commit_context=commit_context,
                )
            )
            current_file = None
            current_diff_lines = []
            current_code_lines = []

        for raw_line in diff_text.splitlines():
            header_match = DIFF_HEADER_PATTERN.match(raw_line)
            if header_match:
                flush_current()
                current_file = header_match.group(2)
                current_diff_lines.append(raw_line)
                continue

            if current_file is None:
                continue

            current_diff_lines.append(raw_line)
            if raw_line.startswith(("+++", "---", "@@")):
                continue
            if raw_line.startswith(("+", "-", " ")):
                current_code_lines.append(raw_line[1:])

        flush_current()
        return evidences

    def _flush_section(
        self,
        *,
        sections: list[PaperSection],
        title: str,
        level: int,
        page_number: int,
        order: int,
        paragraph_buffer: list[str],
    ) -> None:
        """把累计段落落成章节。"""

        content = "\n\n".join(item.strip() for item in paragraph_buffer if item.strip()).strip()
        if not content:
            return

        sections.append(
            PaperSection(
                title=title,
                content=content,
                level=level,
                page_number=page_number,
                order=order,
            )
        )

    def _infer_section_level(self, block: PDFBlock) -> int:
        """根据标题形态推断章节层级。"""

        heading_match = HEADING_NUMBER_PATTERN.match(block.text)
        if heading_match:
            return heading_match.group(1).count(".") + 1
        if block.text.startswith(("第", "Chapter", "SECTION", "Section")):
            return 1
        return 2 if block.font_size < 16 else 1

    def _extract_symbols(self, text: str) -> list[str]:
        """抽取代码标识符。"""

        return [token for token in self._tokenize_text(text) if len(token) >= 2]

    def _tokenize_text(self, text: str) -> list[str]:
        """切词时优先保留标识符和数字，给语义对齐增强策略打底。"""

        return [token.lower() for token in IDENTIFIER_PATTERN.findall(text)]

    def _build_document_frequencies(self, documents: list[list[str]]) -> Counter[str]:
        """统计每个词出现在多少个章节里。"""

        frequencies: Counter[str] = Counter()
        for document in documents:
            frequencies.update(set(document))
        return frequencies

    def _bm25_score(
        self,
        *,
        query_tokens: list[str],
        document_tokens: list[str],
        avg_doc_len: float,
        document_count: int,
        doc_frequencies: Counter[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """用轻量 BM25 评分召回候选章节。"""

        if not query_tokens or not document_tokens:
            return 0.0

        document_counter = Counter(document_tokens)
        document_len = len(document_tokens)
        score = 0.0
        for token in query_tokens:
            frequency = document_counter[token]
            if frequency == 0:
                continue

            document_frequency = doc_frequencies.get(token, 0)
            idf = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (1 - b + b * (document_len / max(avg_doc_len, 1.0)))
            score += idf * (numerator / denominator)
        return score
