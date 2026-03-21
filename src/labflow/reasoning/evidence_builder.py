"""基于轻量检索、AST 逻辑块与 Jedi 定义追踪的证据构建器。"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from importlib import import_module
from pathlib import Path, PurePosixPath

from labflow.clients.llm_client import LLMClient
from labflow.parsers.git_repo_parser import GitRepoParseResult, SourceFile
from labflow.parsers.pdf_parser import PDFBlock, PDFParseResult
from labflow.reasoning.models import (
    AlignmentCandidate,
    CodeEvidence,
    CodeSemanticSummary,
    PaperSection,
)

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")
DIFF_HEADER_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+)$")
HEADING_NUMBER_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)")
SYMBOL_DEFINITION_PATTERN = re.compile(
    r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
SYMBOL_CALL_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")
OPERATOR_SIGNAL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("softmax", ("softmax", "f.softmax", "nn.softmax")),
    ("attention", ("attention", "attn", "q_proj", "k_proj", "v_proj")),
    ("normalization", ("layernorm", "batchnorm", "norm(")),
    ("fusion", ("concat", "cat(", "stack(", "fuse", "fusion")),
    ("reshape", ("reshape", "view(", "permute", "transpose", "flatten")),
    ("dropout", ("dropout", "f.dropout")),
)
SHAPE_SIGNAL_MARKERS = (
    "shape",
    "size(",
    "reshape",
    "view(",
    "permute",
    "transpose",
    "flatten",
    "unsqueeze",
    "squeeze",
    "dim=",
    "dim =",
    "head",
    "heads",
)
PYTHON_KEYWORDS = {
    "if",
    "for",
    "while",
    "return",
    "print",
    "len",
    "range",
    "list",
    "dict",
    "set",
    "tuple",
    "super",
}


class EvidenceBuilder:
    """我负责把仓库切成 Agent 真能追逻辑的证据单元。"""

    def build_project_structure(
        self,
        repo_result: GitRepoParseResult,
        *,
        max_items: int = 60,
    ) -> str:
        source_files = repo_result.source_files[:max_items]
        return self.build_project_structure_from_evidences(
            tuple(
                CodeEvidence(
                    file_name=source_file.relative_path,
                    code_snippet="",
                    related_git_diff="",
                    symbols=(),
                    commit_context=(),
                )
                for source_file in source_files
            )
        )

    def build_project_structure_from_evidences(
        self,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        max_items: int = 60,
    ) -> str:
        unique_paths = list(dict.fromkeys(evidence.file_name for evidence in code_evidences))
        if not unique_paths:
            return "当前代码目录为空。"
        return "\n".join(" / ".join(path.split("/")) for path in unique_paths[:max_items])

    def build_focus_sections(self, pdf_result: PDFParseResult) -> tuple[PaperSection, ...]:
        focus_sections: list[PaperSection] = []
        current_title = "未命名章节"
        current_level = 1
        paragraph_blocks: list[PDFBlock] = []

        for block in pdf_result.blocks:
            if block.kind == "title":
                self._flush_focus_section_group(
                    focus_sections=focus_sections,
                    current_title=current_title,
                    current_level=current_level,
                    paragraph_blocks=paragraph_blocks,
                )
                current_title = block.text
                current_level = self._infer_section_level(block)
                paragraph_blocks = []
                continue

            if not block.text.strip():
                continue

            if paragraph_blocks and not self._should_merge_focus_block(paragraph_blocks[-1], block):
                self._flush_focus_section_group(
                    focus_sections=focus_sections,
                    current_title=current_title,
                    current_level=current_level,
                    paragraph_blocks=paragraph_blocks,
                )
                paragraph_blocks = []
            paragraph_blocks.append(block)

        self._flush_focus_section_group(
            focus_sections=focus_sections,
            current_title=current_title,
            current_level=current_level,
            paragraph_blocks=paragraph_blocks,
        )

        return tuple(focus_sections)

    def _flush_focus_section_group(
        self,
        *,
        focus_sections: list[PaperSection],
        current_title: str,
        current_level: int,
        paragraph_blocks: list[PDFBlock],
    ) -> None:
        if not paragraph_blocks:
            return
        merged_content = " ".join(
            block.text.strip() for block in paragraph_blocks if block.text.strip()
        )
        merged_content = " ".join(merged_content.split())
        first_block = paragraph_blocks[0]
        focus_sections.append(
            PaperSection(
                title=current_title,
                content=merged_content,
                level=current_level,
                page_number=first_block.page_number,
                order=first_block.order,
                block_orders=tuple(block.order for block in paragraph_blocks),
            )
        )

    def _should_merge_focus_block(self, previous_block: PDFBlock, current_block: PDFBlock) -> bool:
        if self._is_same_page_same_column_continuation(previous_block, current_block):
            return True
        if self._is_cross_column_continuation(previous_block, current_block):
            return True
        if self._is_cross_page_continuation(previous_block, current_block):
            return True
        return False

    def _is_same_page_same_column_continuation(
        self, previous_block: PDFBlock, current_block: PDFBlock
    ) -> bool:
        if previous_block.page_number != current_block.page_number:
            return False
        previous_gap = current_block.bbox[1] - previous_block.bbox[3]
        max_gap = max(previous_block.font_size, current_block.font_size) * 1.8
        if previous_gap > max_gap:
            return False
        if not self._is_same_column(previous_block, current_block):
            return False
        return True

    def _is_cross_page_continuation(
        self, previous_block: PDFBlock, current_block: PDFBlock
    ) -> bool:
        if current_block.page_number - previous_block.page_number != 1:
            return False
        if not self._is_same_column(previous_block, current_block):
            return False
        if not self._looks_like_continuation(previous_block.text, current_block.text):
            return False
        if previous_block.page_height <= 0 or current_block.page_height <= 0:
            return False
        previous_bottom_ratio = previous_block.bbox[3] / previous_block.page_height
        current_top_ratio = current_block.bbox[1] / current_block.page_height
        if previous_bottom_ratio < 0.72:
            return False
        if current_top_ratio > 0.2:
            return False
        if not self._is_similar_block_width(previous_block, current_block):
            return False
        return True

    def _is_cross_column_continuation(
        self, previous_block: PDFBlock, current_block: PDFBlock
    ) -> bool:
        if previous_block.page_number != current_block.page_number:
            return False
        if self._is_same_column(previous_block, current_block):
            return False
        if not self._looks_like_continuation(previous_block.text, current_block.text):
            return False
        if previous_block.page_width <= 0 or current_block.page_width <= 0:
            return False
        previous_bottom_ratio = previous_block.bbox[3] / previous_block.page_height
        current_top_ratio = current_block.bbox[1] / current_block.page_height
        left_shift = current_block.bbox[0] - previous_block.bbox[0]
        if left_shift < previous_block.page_width * 0.12:
            return False
        if previous_bottom_ratio < 0.55:
            return False
        if current_top_ratio > 0.32:
            return False
        if not self._is_similar_block_width(previous_block, current_block):
            return False
        return True

    def _is_same_column(self, previous_block: PDFBlock, current_block: PDFBlock) -> bool:
        left_offset = abs(current_block.bbox[0] - previous_block.bbox[0])
        return left_offset <= max(previous_block.font_size, current_block.font_size) * 2.5

    def _is_similar_block_width(self, previous_block: PDFBlock, current_block: PDFBlock) -> bool:
        previous_width = max(previous_block.bbox[2] - previous_block.bbox[0], 1.0)
        current_width = max(current_block.bbox[2] - current_block.bbox[0], 1.0)
        width_ratio = min(previous_width, current_width) / max(previous_width, current_width)
        return width_ratio >= 0.72

    def _looks_like_continuation(self, previous_text: str, current_text: str) -> bool:
        previous_tail = previous_text.rstrip()
        current_head = current_text.lstrip()
        if not previous_tail or not current_head:
            return False
        if previous_tail.endswith("-"):
            return True
        if previous_tail[-1] not in ".!?;:。！？；：":
            return True
        first_char = current_head[0]
        return first_char.islower() or first_char.isdigit()

    def build_paper_sections(self, pdf_result: PDFParseResult) -> tuple[PaperSection, ...]:
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
        commit_context = tuple(commit.summary for commit in repo_result.recent_commits)
        repo_root = Path(repo_result.repo_path)

        if repo_result.source_files:
            diff_lookup = self._build_diff_lookup(repo_result.working_tree_diff)
            evidences = self._build_source_file_evidences(
                repo_result.source_files,
                repo_root=repo_root,
                commit_context=commit_context,
                diff_lookup=diff_lookup,
            )
            if evidences:
                return tuple(evidences)

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

    def build_semantic_index(
        self,
        repo_result: GitRepoParseResult,
        *,
        llm_client: LLMClient | None = None,
    ) -> tuple[CodeSemanticSummary, ...]:
        return self.build_semantic_index_from_evidences(
            self.build_code_evidences(repo_result),
            llm_client=llm_client,
        )

    def build_semantic_index_from_evidences(
        self,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        llm_client: LLMClient | None = None,
    ) -> tuple[CodeSemanticSummary, ...]:
        summaries: list[CodeSemanticSummary] = []
        for evidence in code_evidences:
            summaries.append(self._summarize_code_evidence(evidence, llm_client=llm_client))
        return tuple(summaries)

    def retrieve_semantic_candidates(
        self,
        paper_section: PaperSection,
        semantic_index: tuple[CodeSemanticSummary, ...],
        *,
        top_k: int = 4,
    ) -> tuple[AlignmentCandidate, ...]:
        if not semantic_index:
            return ()

        documents = [self._tokenize_text(summary.search_text) for summary in semantic_index]
        doc_frequencies = self._build_document_frequencies(documents)
        avg_doc_len = sum(len(tokens) for tokens in documents) / max(len(documents), 1)
        query_tokens = self._tokenize_text(paper_section.combined_text)

        scored_candidates: list[AlignmentCandidate] = []
        for summary, tokens in zip(semantic_index, documents, strict=False):
            score = self._bm25_score(
                query_tokens=query_tokens,
                document_tokens=tokens,
                avg_doc_len=avg_doc_len,
                document_count=len(documents),
                doc_frequencies=doc_frequencies,
            )
            scored_candidates.append(
                AlignmentCandidate(
                    paper_section=paper_section,
                    code_evidence=summary.code_evidence,
                    retrieval_score=round(score, 4),
                )
            )

        scored_candidates.sort(key=lambda item: item.retrieval_score, reverse=True)
        return tuple(scored_candidates[:top_k])

    def trace_related_candidates(
        self,
        paper_section: PaperSection,
        semantic_index: tuple[CodeSemanticSummary, ...],
        *,
        trace_symbols: tuple[str, ...],
        seen_candidate_ids: set[str] | None = None,
        limit: int = 4,
    ) -> tuple[AlignmentCandidate, ...]:
        normalized_symbols = {symbol.lower() for symbol in trace_symbols if symbol}
        if not normalized_symbols:
            return ()

        seen_candidate_ids = seen_candidate_ids or set()
        traced_candidates: list[AlignmentCandidate] = []
        for summary in semantic_index:
            if summary.identity in seen_candidate_ids:
                continue

            defined_symbols = {symbol.lower() for symbol in summary.defined_symbols}
            called_symbols = {symbol.lower() for symbol in summary.called_symbols}
            anchor_terms = {symbol.lower() for symbol in summary.anchor_terms}
            overlap = normalized_symbols & (defined_symbols | called_symbols | anchor_terms)
            if not overlap:
                continue

            traced_candidates.append(
                AlignmentCandidate(
                    paper_section=paper_section,
                    code_evidence=summary.code_evidence,
                    retrieval_score=round(0.25 + 0.12 * len(overlap), 4),
                )
            )
            if len(traced_candidates) >= limit:
                break

        traced_candidates.sort(key=lambda item: item.retrieval_score, reverse=True)
        return tuple(traced_candidates[:limit])

    def build_alignment_candidates(
        self,
        pdf_result: PDFParseResult,
        repo_result: GitRepoParseResult,
        top_k: int = 2,
    ) -> tuple[AlignmentCandidate, ...]:
        paper_sections = self.build_paper_sections(pdf_result)
        code_evidences = self.build_code_evidences(repo_result)
        return self.build_alignment_candidates_from_inputs(
            paper_sections,
            code_evidences,
            top_k=top_k,
        )

    def build_alignment_candidates_from_inputs(
        self,
        paper_sections: tuple[PaperSection, ...],
        code_evidences: tuple[CodeEvidence, ...],
        top_k: int = 2,
    ) -> tuple[AlignmentCandidate, ...]:
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

    def read_logic_block(
        self,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        path: str,
        line_start: int,
        line_end: int,
    ) -> tuple[str, tuple[str, ...]]:
        evidence = self._find_best_evidence(
            code_evidences,
            path=path,
            line_start=line_start,
            line_end=line_end,
        )
        if evidence is None:
            return "没有找到指定代码段。", ()

        docstring = evidence.docstring.strip() or "无"
        summary = (
            f"文件: {evidence.file_name}\n"
            f"逻辑块类型: {evidence.block_type}\n"
            f"符号: {evidence.symbol_name or '未命名逻辑块'}\n"
            f"范围: L{evidence.start_line}-L{evidence.end_line}\n"
            f"Docstring: {docstring}\n\n"
            f"代码:\n{evidence.code_snippet}"
        )
        return summary, (self._candidate_id_from_evidence(evidence),)

    def find_definition_candidate(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        symbol: str,
        file_path: str,
        line: int,
        column: int,
    ) -> tuple[str, tuple[AlignmentCandidate, ...]]:
        base_evidence = self._find_best_evidence(
            code_evidences,
            path=file_path,
            line_start=line,
            line_end=line,
        )
        if base_evidence is None:
            return "没有找到用于跳转定义的起始代码块。", ()

        if not base_evidence.absolute_path or not base_evidence.source_content:
            fallback = self._find_symbol_candidate_by_name(
                paper_section,
                code_evidences,
                symbol=symbol,
            )
            if fallback is None:
                return "当前代码块缺少可供 Jedi 跳转的源码上下文。", ()
            observation = self._build_definition_observation(
                symbol,
                fallback.code_evidence,
                reason="Jedi 上下文不足，已退回到仓库内符号搜索。",
            )
            return observation, (fallback,)

        target_evidence = self._resolve_definition_with_jedi(
            paper_section,
            code_evidences,
            symbol=symbol,
            file_path=file_path,
            line=line,
            column=column,
            base_evidence=base_evidence,
        )
        if target_evidence is None:
            fallback = self._find_symbol_candidate_by_name(
                paper_section,
                code_evidences,
                symbol=symbol,
            )
            if fallback is None:
                return f"没有在仓库内追踪到符号 `{symbol}` 的定义。", ()
            observation = self._build_definition_observation(
                symbol,
                fallback.code_evidence,
                reason="Jedi 未给出稳定结果，已退回到仓库内符号搜索。",
            )
            return observation, (fallback,)

        observation = self._build_definition_observation(symbol, target_evidence.code_evidence)
        return observation, (target_evidence,)

    def _build_source_file_evidences(
        self,
        source_files: tuple[SourceFile, ...],
        *,
        repo_root: Path,
        commit_context: tuple[str, ...],
        diff_lookup: dict[str, str],
    ) -> list[CodeEvidence]:
        evidences: list[CodeEvidence] = []
        for source_file in source_files:
            if source_file.language == "python":
                chunks = self._chunk_python_source_file(source_file, repo_root=repo_root)
            else:
                chunks = self._chunk_plain_source_file(source_file, repo_root=repo_root)

            if not chunks:
                chunks = self._chunk_plain_source_file(source_file, repo_root=repo_root)

            for evidence in chunks:
                evidences.append(
                    CodeEvidence(
                        file_name=evidence.file_name,
                        code_snippet=evidence.code_snippet,
                        related_git_diff=diff_lookup.get(evidence.file_name, ""),
                        symbols=evidence.symbols,
                        commit_context=commit_context,
                        start_line=evidence.start_line,
                        end_line=evidence.end_line,
                        language=evidence.language,
                        absolute_path=evidence.absolute_path,
                        source_content=evidence.source_content,
                        symbol_name=evidence.symbol_name,
                        parent_symbol=evidence.parent_symbol,
                        block_type=evidence.block_type,
                        docstring=evidence.docstring,
                    )
                )
        return evidences

    def _chunk_python_source_file(
        self,
        source_file: SourceFile,
        *,
        repo_root: Path,
        max_chunks_per_file: int = 12,
    ) -> list[CodeEvidence]:
        absolute_path = str((repo_root / source_file.relative_path).resolve())
        lines = source_file.content.splitlines()
        if not lines:
            return []

        try:
            module = ast.parse(source_file.content, filename=source_file.relative_path)
        except SyntaxError:
            return self._chunk_plain_source_file(
                source_file,
                repo_root=repo_root,
                max_chunks_per_file=max_chunks_per_file,
            )

        chunks: list[CodeEvidence] = []
        seen_ranges: set[tuple[int, int, str]] = set()
        top_level_nodes = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if top_level_nodes and top_level_nodes[0].lineno > 1:
            intro_end = min(top_level_nodes[0].lineno - 1, 24)
            intro_text = "\n".join(lines[:intro_end]).strip()
            if intro_text:
                chunks.append(
                    self._create_code_evidence(
                        relative_path=source_file.relative_path,
                        absolute_path=absolute_path,
                        source_content=source_file.content,
                        language=source_file.language,
                        start_line=1,
                        end_line=intro_end,
                        code_snippet=intro_text,
                        symbol_name="module_intro",
                        parent_symbol="",
                        block_type="module",
                        docstring="",
                    )
                )

        for node in top_level_nodes:
            self._append_ast_node_evidence(
                chunks,
                seen_ranges,
                source_file=source_file,
                absolute_path=absolute_path,
                node=node,
                parent_symbol="",
            )
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._append_ast_node_evidence(
                            chunks,
                            seen_ranges,
                            source_file=source_file,
                            absolute_path=absolute_path,
                            node=child,
                            parent_symbol=node.name,
                        )
            if len(chunks) >= max_chunks_per_file:
                break

        return chunks[:max_chunks_per_file]

    def _append_ast_node_evidence(
        self,
        chunks: list[CodeEvidence],
        seen_ranges: set[tuple[int, int, str]],
        *,
        source_file: SourceFile,
        absolute_path: str,
        node: ast.AST,
        parent_symbol: str,
    ) -> None:
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        if end_line < start_line:
            end_line = start_line

        block_type = self._infer_block_type(node, parent_symbol=parent_symbol)
        symbol_name = getattr(node, "name", "") if hasattr(node, "name") else ""
        if parent_symbol and symbol_name:
            symbol_name = f"{parent_symbol}.{symbol_name}"
        range_key = (start_line, end_line, symbol_name or block_type)
        if range_key in seen_ranges:
            return
        seen_ranges.add(range_key)

        snippet = "\n".join(source_file.content.splitlines()[start_line - 1 : end_line]).strip()
        if not snippet:
            return

        docstring = ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node) or ""

        chunks.append(
            self._create_code_evidence(
                relative_path=source_file.relative_path,
                absolute_path=absolute_path,
                source_content=source_file.content,
                language=source_file.language,
                start_line=start_line,
                end_line=end_line,
                code_snippet=snippet,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
                block_type=block_type,
                docstring=docstring,
            )
        )

    def _chunk_plain_source_file(
        self,
        source_file: SourceFile,
        *,
        repo_root: Path,
        max_chunk_lines: int = 48,
        max_chunks_per_file: int = 8,
    ) -> list[CodeEvidence]:
        lines = source_file.content.splitlines()
        absolute_path = str((repo_root / source_file.relative_path).resolve())
        chunks: list[CodeEvidence] = []
        for start_index in range(0, len(lines), max_chunk_lines):
            start_line = start_index + 1
            end_line = min(start_index + max_chunk_lines, len(lines))
            snippet = "\n".join(lines[start_index:end_line]).strip()
            if not snippet:
                continue
            chunks.append(
                self._create_code_evidence(
                    relative_path=source_file.relative_path,
                    absolute_path=absolute_path,
                    source_content=source_file.content,
                    language=source_file.language,
                    start_line=start_line,
                    end_line=end_line,
                    code_snippet=snippet,
                    symbol_name="",
                    parent_symbol="",
                    block_type="snippet",
                    docstring="",
                )
            )
            if len(chunks) >= max_chunks_per_file:
                break
        return chunks

    def _create_code_evidence(
        self,
        *,
        relative_path: str,
        absolute_path: str,
        source_content: str,
        language: str,
        start_line: int,
        end_line: int,
        code_snippet: str,
        symbol_name: str,
        parent_symbol: str,
        block_type: str,
        docstring: str,
    ) -> CodeEvidence:
        operator_signals = self._extract_operator_signals(code_snippet)
        shape_signals = self._extract_shape_signals(code_snippet)
        raw_symbols = [
            symbol_name,
            parent_symbol,
            *self._extract_defined_symbols(code_snippet),
            *self._extract_called_symbols(code_snippet),
            *operator_signals,
            *shape_signals,
            *self._extract_symbols(code_snippet),
        ]
        symbols = tuple(dict.fromkeys(item for item in raw_symbols if item))
        return CodeEvidence(
            file_name=relative_path,
            code_snippet=code_snippet,
            related_git_diff="",
            symbols=symbols,
            commit_context=(),
            start_line=start_line,
            end_line=end_line,
            language=language,
            absolute_path=absolute_path,
            source_content=source_content,
            symbol_name=symbol_name,
            parent_symbol=parent_symbol,
            block_type=block_type,
            docstring=docstring,
        )

    def _parse_diff_evidences(
        self,
        diff_text: str,
        commit_context: tuple[str, ...],
    ) -> list[CodeEvidence]:
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

    def _build_diff_lookup(self, diff_text: str) -> dict[str, str]:
        if not diff_text.strip():
            return {}

        diff_lookup: dict[str, str] = {}
        current_file: str | None = None
        current_lines: list[str] = []

        def flush_current() -> None:
            nonlocal current_file, current_lines
            if current_file and current_lines:
                diff_lookup[current_file] = "\n".join(current_lines).strip()
            current_file = None
            current_lines = []

        for raw_line in diff_text.splitlines():
            header_match = DIFF_HEADER_PATTERN.match(raw_line)
            if header_match:
                flush_current()
                current_file = PurePosixPath(header_match.group(2)).as_posix()
                current_lines.append(raw_line)
                continue
            if current_file is not None:
                current_lines.append(raw_line)

        flush_current()
        return diff_lookup

    def _summarize_code_evidence(
        self,
        evidence: CodeEvidence,
        *,
        llm_client: LLMClient | None = None,
    ) -> CodeSemanticSummary:
        fallback_defined_symbols = tuple(self._extract_defined_symbols(evidence.code_snippet))
        fallback_called_symbols = tuple(self._extract_called_symbols(evidence.code_snippet))
        fallback_anchor_terms = tuple(
            dict.fromkeys(
                token
                for token in (
                    evidence.symbol_name,
                    evidence.parent_symbol,
                    *evidence.symbols,
                    *fallback_defined_symbols,
                    *fallback_called_symbols,
                )
                if token and len(token) >= 2
            )
        )

        if llm_client is None:
            return self._build_fallback_semantic_summary(
                evidence,
                defined_symbols=fallback_defined_symbols,
                called_symbols=fallback_called_symbols,
                anchor_terms=fallback_anchor_terms,
            )

        try:
            payload = llm_client.generate_json(
                system_prompt=self._build_semantic_summary_system_prompt(),
                user_prompt=self._build_semantic_summary_user_prompt(evidence),
                temperature=0.0,
                max_tokens=700,
            )
        except RuntimeError:
            payload = None

        if not isinstance(payload, dict):
            return self._build_fallback_semantic_summary(
                evidence,
                defined_symbols=fallback_defined_symbols,
                called_symbols=fallback_called_symbols,
                anchor_terms=fallback_anchor_terms,
            )

        responsibilities = self._normalize_string_list(payload.get("responsibilities"))
        defined_symbols = tuple(
            self._merge_symbol_lists(
                fallback_defined_symbols,
                self._normalize_string_list(payload.get("defined_symbols")),
            )
        )
        called_symbols = tuple(
            self._merge_symbol_lists(
                fallback_called_symbols,
                self._normalize_string_list(payload.get("called_symbols")),
            )
        )
        anchor_terms = tuple(
            self._merge_symbol_lists(
                fallback_anchor_terms,
                self._normalize_string_list(payload.get("anchor_terms")),
            )
        )
        summary = str(payload.get("summary", "")).strip() or self._build_fallback_summary_text(
            evidence,
            defined_symbols=defined_symbols,
            called_symbols=called_symbols,
        )

        return CodeSemanticSummary(
            code_evidence=evidence,
            summary=summary,
            responsibilities=responsibilities or (summary,),
            defined_symbols=defined_symbols,
            called_symbols=called_symbols,
            anchor_terms=anchor_terms,
        )

    def _build_fallback_semantic_summary(
        self,
        evidence: CodeEvidence,
        *,
        defined_symbols: tuple[str, ...],
        called_symbols: tuple[str, ...],
        anchor_terms: tuple[str, ...],
    ) -> CodeSemanticSummary:
        summary = self._build_fallback_summary_text(
            evidence,
            defined_symbols=defined_symbols,
            called_symbols=called_symbols,
        )
        responsibilities = (
            f"该片段位于 {evidence.file_name} 的 L{evidence.start_line}-L{evidence.end_line}",
            "需要结合真实调用链继续核对实现逻辑。",
        )
        return CodeSemanticSummary(
            code_evidence=evidence,
            summary=summary,
            responsibilities=responsibilities,
            defined_symbols=defined_symbols,
            called_symbols=called_symbols,
            anchor_terms=anchor_terms,
        )

    def _build_fallback_summary_text(
        self,
        evidence: CodeEvidence,
        *,
        defined_symbols: tuple[str, ...],
        called_symbols: tuple[str, ...],
    ) -> str:
        summary_parts = [
            f"{evidence.file_name} 的 {evidence.block_type} 逻辑块",
            f"覆盖 L{evidence.start_line}-L{evidence.end_line}",
        ]
        if evidence.symbol_name:
            summary_parts.append(f"核心符号是 {evidence.symbol_name}")
        if defined_symbols:
            summary_parts.append(f"定义了 {', '.join(defined_symbols[:4])}")
        if called_symbols:
            summary_parts.append(f"调用了 {', '.join(called_symbols[:4])}")
        return "，".join(summary_parts) + "。"

    def _build_semantic_summary_system_prompt(self) -> str:
        return """
你是仓库语义索引构建器。你的任务是阅读一个代码片段，提炼它在系统中的逻辑职责，
方便后续 Agent 做调用链追踪。不要重复代码原文，不要泛泛而谈。
只输出 JSON，字段包括：
- summary: 中文一句话总结这段代码在做什么
- responsibilities: 中文短句列表，列出 2-4 个职责
- defined_symbols: 这段代码显式定义的函数、类或核心变量名列表
- called_symbols: 这段代码调用或依赖的函数 / 模块名列表
- anchor_terms: 后续检索时可用的机制词、算法词、参数词列表
""".strip()

    def _build_semantic_summary_user_prompt(self, evidence: CodeEvidence) -> str:
        commit_context = "\n".join(evidence.commit_context) if evidence.commit_context else "无"
        return f"""
【文件】{evidence.file_name}
【代码范围】L{evidence.start_line}-L{evidence.end_line}
【逻辑块类型】{evidence.block_type}
【符号】{evidence.symbol_name or "未命名逻辑块"}
【代码片段】
{evidence.code_snippet}

【关联 Diff】
{evidence.related_git_diff or "无"}

【最近提交上下文】
{commit_context}
""".strip()

    def _resolve_definition_with_jedi(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        symbol: str,
        file_path: str,
        line: int,
        column: int,
        base_evidence: CodeEvidence,
    ) -> AlignmentCandidate | None:
        jedi_module = self._load_jedi_module()
        source_content = base_evidence.source_content
        actual_column = self._normalize_definition_column(
            source_content,
            symbol=symbol,
            line=line,
            column=column,
        )

        try:
            script = jedi_module.Script(code=source_content, path=base_evidence.absolute_path)
            definitions = script.goto(
                line=line,
                column=actual_column,
                follow_imports=True,
                follow_builtin_imports=False,
            )
            if not definitions:
                definitions = script.infer(line=line, column=actual_column)
        except Exception:  # noqa: BLE001
            return None

        repo_root = (
            Path(base_evidence.absolute_path)
            .resolve()
            .parents[len(PurePosixPath(file_path).parts) - 1]
        )
        for definition in definitions:
            module_path = getattr(definition, "module_path", None)
            target_line = getattr(definition, "line", None)
            if module_path is None or target_line is None:
                continue

            try:
                relative_path = Path(module_path).resolve().relative_to(repo_root).as_posix()
            except ValueError:
                continue

            target_evidence = self._find_best_evidence(
                code_evidences,
                path=relative_path,
                line_start=target_line,
                line_end=target_line,
            )
            if target_evidence is None:
                continue

            return AlignmentCandidate(
                paper_section=paper_section,
                code_evidence=target_evidence,
                retrieval_score=max(0.72, min(0.95, target_evidence.start_line / 1000 + 0.72)),
            )
        return None

    def _build_definition_observation(
        self,
        symbol: str,
        evidence: CodeEvidence,
        *,
        reason: str | None = None,
    ) -> str:
        prefix = f"已追踪到 `{symbol}` 的定义"
        if reason:
            prefix = f"{reason} 最终命中了 `{symbol}` 的定义"
        return (
            f"{prefix}：{evidence.file_name} "
            f"L{evidence.start_line}-L{evidence.end_line}，"
            f"逻辑块是 {evidence.symbol_name or evidence.block_type}。"
        )

    def _find_symbol_candidate_by_name(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        symbol: str,
    ) -> AlignmentCandidate | None:
        normalized = symbol.strip().lower()
        for evidence in code_evidences:
            symbol_pool = {item.lower() for item in evidence.symbols if item}
            symbol_match = evidence.symbol_name.lower() if evidence.symbol_name else ""
            if normalized == symbol_match or normalized in symbol_pool:
                return AlignmentCandidate(
                    paper_section=paper_section,
                    code_evidence=evidence,
                    retrieval_score=0.68,
                )
        return None

    def _find_best_evidence(
        self,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        path: str,
        line_start: int,
        line_end: int,
    ) -> CodeEvidence | None:
        exact_matches = [
            evidence
            for evidence in code_evidences
            if evidence.file_name == path
            and evidence.start_line <= line_start
            and evidence.end_line >= line_end
        ]
        if exact_matches:
            return min(exact_matches, key=lambda item: item.end_line - item.start_line)

        for evidence in code_evidences:
            if evidence.file_name == path:
                return evidence
        return None

    def _candidate_id_from_evidence(self, evidence: CodeEvidence) -> str:
        return f"{evidence.file_name}:{evidence.start_line}-{evidence.end_line}"

    def _load_jedi_module(self):
        try:
            jedi_module = import_module("jedi")
        except ModuleNotFoundError as exc:
            raise RuntimeError("当前环境缺少 jedi，先安装依赖后再执行跨文件追踪。") from exc
        cache_dir = Path(".tmp/jedi-cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(jedi_module, "settings"):
            jedi_module.settings.cache_directory = str(cache_dir.resolve())
        return jedi_module

    def _normalize_definition_column(
        self,
        source_content: str,
        *,
        symbol: str,
        line: int,
        column: int,
    ) -> int:
        lines = source_content.splitlines()
        if 1 <= line <= len(lines):
            raw_line = lines[line - 1]
            if 0 <= column < len(raw_line):
                return column
            symbol_index = raw_line.find(symbol)
            if symbol_index >= 0:
                return symbol_index
            return max(0, min(len(raw_line) - 1, column))
        return 0

    def _extract_defined_symbols(self, text: str) -> list[str]:
        return [symbol for symbol in SYMBOL_DEFINITION_PATTERN.findall(text) if symbol]

    def _extract_called_symbols(self, text: str) -> list[str]:
        called_symbols: list[str] = []
        for symbol in SYMBOL_CALL_PATTERN.findall(text):
            if symbol in PYTHON_KEYWORDS:
                continue
            if symbol not in called_symbols:
                called_symbols.append(symbol)
        return called_symbols

    def _extract_operator_signals(self, text: str) -> tuple[str, ...]:
        lowered = text.lower()
        signals: list[str] = []
        for name, patterns in OPERATOR_SIGNAL_PATTERNS:
            if any(pattern in lowered for pattern in patterns):
                signals.append(name)
        return tuple(signals)

    def _extract_shape_signals(self, text: str) -> tuple[str, ...]:
        lowered = text.lower()
        return tuple(marker for marker in SHAPE_SIGNAL_MARKERS if marker in lowered)

    def _merge_symbol_lists(self, *groups: tuple[str, ...] | list[str]) -> list[str]:
        merged: list[str] = []
        for group in groups:
            for item in group:
                normalized = item.strip()
                if not normalized or normalized in merged:
                    continue
                merged.append(normalized)
        return merged

    def _normalize_string_list(self, value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if not text or text in normalized:
                continue
            normalized.append(text)
        return tuple(normalized)

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
        heading_match = HEADING_NUMBER_PATTERN.match(block.text)
        if heading_match:
            return heading_match.group(1).count(".") + 1
        if block.text.startswith(("第", "Chapter", "SECTION", "Section")):
            return 1
        return 2 if block.font_size < 16 else 1

    def _extract_symbols(self, text: str) -> list[str]:
        return [token for token in self._tokenize_text(text) if len(token) >= 2]

    def _tokenize_text(self, text: str) -> list[str]:
        return [token.lower() for token in IDENTIFIER_PATTERN.findall(text)]

    def _build_document_frequencies(self, documents: list[list[str]]) -> Counter[str]:
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

    def _infer_block_type(self, node: ast.AST, *, parent_symbol: str) -> str:
        if isinstance(node, ast.ClassDef):
            return "class"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "method" if parent_symbol else "function"
        return "snippet"
