"""基于轻量检索与语义索引的证据构建器。"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import PurePosixPath

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
    """我把仓库切成可追踪的语义单元，让 Agent 能从召回走向逻辑追踪。"""

    def build_project_structure(
        self,
        repo_result: GitRepoParseResult,
        *,
        max_items: int = 60,
    ) -> str:
        """初始化阶段只构建文件树，不触发任何 LLM 行为。"""

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
        """从代码证据提取稳定文件树文本。"""

        unique_paths = list(dict.fromkeys(evidence.file_name for evidence in code_evidences))
        if not unique_paths:
            return "当前代码目录为空。"
        return "\n".join(" / ".join(path.split("/")) for path in unique_paths[:max_items])

    def build_focus_sections(self, pdf_result: PDFParseResult) -> tuple[PaperSection, ...]:
        """把每一个正文块转成可点击的阅读焦点，兼顾段落粒度和章节上下文。"""

        focus_sections: list[PaperSection] = []
        current_title = "未命名章节"
        current_level = 1

        for block in pdf_result.blocks:
            if block.kind == "title":
                current_title = block.text
                current_level = self._infer_section_level(block)
                continue

            if not block.text.strip():
                continue

            focus_sections.append(
                PaperSection(
                    title=current_title,
                    content=block.text,
                    level=current_level,
                    page_number=block.page_number,
                    order=block.order,
                )
            )

        return tuple(focus_sections)

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
        """把 Git diff 和源文件切成可展示、可追踪的代码证据。"""

        commit_context = tuple(commit.summary for commit in repo_result.recent_commits)
        if repo_result.source_files:
            diff_lookup = self._build_diff_lookup(repo_result.working_tree_diff)
            evidences = self._build_source_file_evidences(
                repo_result.source_files,
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
        """先把仓库扫描成语义摘要，给 Agent 后续循环检索提供稳定索引。"""

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
        """从代码证据生成语义索引条目。"""

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
        """利用代码语义摘要做首轮候选召回。"""

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
        """根据调用链或符号链做二次追踪，帮助 Agent 从片段走向实现链路。"""

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
        """兼容原有批量候选召回接口。"""

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
        """对任意章节/证据输入执行轻量召回。"""

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

    def _build_source_file_evidences(
        self,
        source_files: tuple[SourceFile, ...],
        *,
        commit_context: tuple[str, ...],
        diff_lookup: dict[str, str],
    ) -> list[CodeEvidence]:
        """把源代码文件切成可读片段，给右栏检查器提供带行号的真实代码。"""

        evidences: list[CodeEvidence] = []
        for source_file in source_files:
            chunks = self._chunk_source_file(source_file)
            for start_line, end_line, code_snippet in chunks:
                symbols = tuple(
                    self._extract_symbols(f"{source_file.relative_path}\n{code_snippet}")
                )
                evidences.append(
                    CodeEvidence(
                        file_name=source_file.relative_path,
                        code_snippet=code_snippet,
                        related_git_diff=diff_lookup.get(source_file.relative_path, ""),
                        symbols=symbols,
                        commit_context=commit_context,
                        start_line=start_line,
                        end_line=end_line,
                        language=source_file.language,
                    )
                )
        return evidences

    def _parse_diff_evidences(
        self,
        diff_text: str,
        commit_context: tuple[str, ...],
    ) -> list[CodeEvidence]:
        """按文件粒度拆 diff。"""

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
        """把 diff 按文件归档，方便代码片段旁边挂关联变更。"""

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

    def _chunk_source_file(
        self,
        source_file: SourceFile,
        *,
        max_chunk_lines: int = 48,
        max_chunks_per_file: int = 8,
    ) -> list[tuple[int, int, str]]:
        """尽量按函数/类边界切块，让人点章节时右栏看到的是完整语义单元。"""

        lines = source_file.content.splitlines()
        if not lines:
            return []

        marker_lines = [
            index
            for index, line in enumerate(lines, start=1)
            if line.lstrip().startswith(("def ", "class ", "async def "))
        ]

        chunks: list[tuple[int, int, str]] = []
        if marker_lines and marker_lines[0] > 1:
            intro_end = min(marker_lines[0] - 1, max_chunk_lines)
            intro_snippet = "\n".join(lines[:intro_end]).strip()
            if intro_snippet:
                chunks.append((1, intro_end, intro_snippet))

        if marker_lines:
            for index, start_line in enumerate(marker_lines):
                next_marker = (
                    marker_lines[index + 1] if index + 1 < len(marker_lines) else len(lines) + 1
                )
                end_line = min(next_marker - 1, start_line + max_chunk_lines - 1, len(lines))
                snippet = "\n".join(lines[start_line - 1 : end_line]).strip()
                if snippet:
                    chunks.append((start_line, end_line, snippet))
                if len(chunks) >= max_chunks_per_file:
                    break
            return chunks[:max_chunks_per_file]

        for start_index in range(0, len(lines), max_chunk_lines):
            start_line = start_index + 1
            end_line = min(start_index + max_chunk_lines, len(lines))
            snippet = "\n".join(lines[start_index:end_line]).strip()
            if snippet:
                chunks.append((start_line, end_line, snippet))
            if len(chunks) >= max_chunks_per_file:
                break
        return chunks

    def _summarize_code_evidence(
        self,
        evidence: CodeEvidence,
        *,
        llm_client: LLMClient | None = None,
    ) -> CodeSemanticSummary:
        """先给代码片段做逻辑摘要，后续 Agent 才能沿调用链继续查。"""

        fallback_defined_symbols = tuple(self._extract_defined_symbols(evidence.code_snippet))
        fallback_called_symbols = tuple(self._extract_called_symbols(evidence.code_snippet))
        fallback_anchor_terms = tuple(
            dict.fromkeys(
                token
                for token in (
                    *evidence.symbols,
                    *fallback_defined_symbols,
                    *fallback_called_symbols,
                )
                if len(token) >= 2
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
            return self._build_fallback_semantic_summary(
                evidence,
                defined_symbols=fallback_defined_symbols,
                called_symbols=fallback_called_symbols,
                anchor_terms=fallback_anchor_terms,
            )

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
        """当模型不可用时，至少保留一个能支持追链路的摘要兜底。"""

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
        """用结构信息拼一个低配逻辑摘要。"""

        summary_parts = [
            f"{evidence.file_name} 的代码片段",
            f"覆盖 L{evidence.start_line}-L{evidence.end_line}",
        ]
        if defined_symbols:
            summary_parts.append(f"定义了 {', '.join(defined_symbols[:4])}")
        if called_symbols:
            summary_parts.append(f"调用了 {', '.join(called_symbols[:4])}")
        return "，".join(summary_parts) + "。"

    def _build_semantic_summary_system_prompt(self) -> str:
        """告诉模型如何把代码切片压成后续检索可用的语义摘要。"""

        return """
你是仓库语义索引构建器。
你的任务是阅读一个代码片段，提炼它在系统中的逻辑职责，方便后续 Agent 做调用链追踪。
不要重复代码原文，不要泛泛而谈。
只输出 JSON，字段包括：
- summary: 中文一句话总结这段代码在做什么
- responsibilities: 中文短句列表，列出 2-4 个职责
- defined_symbols: 这段代码显式定义的函数、类或核心变量名列表
- called_symbols: 这段代码调用或依赖的函数/模块名列表
- anchor_terms: 后续检索时可用的机制词、算法词、参数词列表
""".strip()

    def _build_semantic_summary_user_prompt(self, evidence: CodeEvidence) -> str:
        """为语义索引构建提供上下文。"""

        commit_context = "\n".join(evidence.commit_context) if evidence.commit_context else "无"
        return f"""
【文件】{evidence.file_name}
【代码范围】L{evidence.start_line}-L{evidence.end_line}
【代码片段】
{evidence.code_snippet}

【关联 Diff】
{evidence.related_git_diff or "无"}

【最近提交上下文】
{commit_context}
""".strip()

    def _extract_defined_symbols(self, text: str) -> list[str]:
        """提取定义出来的函数名和类名。"""

        return [symbol for symbol in SYMBOL_DEFINITION_PATTERN.findall(text) if symbol]

    def _extract_called_symbols(self, text: str) -> list[str]:
        """提取潜在调用符号，给二次追踪提供线索。"""

        called_symbols: list[str] = []
        for symbol in SYMBOL_CALL_PATTERN.findall(text):
            if symbol in PYTHON_KEYWORDS:
                continue
            if symbol not in called_symbols:
                called_symbols.append(symbol)
        return called_symbols

    def _merge_symbol_lists(self, *groups: tuple[str, ...] | list[str]) -> list[str]:
        """把多来源符号合并去重。"""

        merged: list[str] = []
        for group in groups:
            for item in group:
                normalized = item.strip()
                if not normalized or normalized in merged:
                    continue
                merged.append(normalized)
        return merged

    def _normalize_string_list(self, value: object) -> tuple[str, ...]:
        """把模型返回的列表字段收敛成稳定字符串元组。"""

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
        """统计每个词出现在多少个文档里。"""

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
        """用轻量 BM25 评分召回候选文档。"""

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
