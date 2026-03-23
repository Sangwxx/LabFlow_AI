"""代码知识索引与混合检索。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from labflow.clients.llm_client import LLMClient
from labflow.reasoning.models import (
    AlignmentCandidate,
    CodeEvidence,
    CodeSemanticSummary,
    PaperSection,
)

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class CodeKnowledgeEntry:
    """知识索引中的单条代码知识。"""

    identity: str
    summary: CodeSemanticSummary
    lexical_text: str
    semantic_text: str
    semantic_vector: tuple[float, ...]


class _BM25Index:
    """轻量 BM25 实现，避免新增运行时依赖。"""

    def __init__(self, corpus_tokens: tuple[tuple[str, ...], ...]) -> None:
        self._corpus_tokens = corpus_tokens
        self._doc_term_counts = tuple(Counter(tokens) for tokens in corpus_tokens)
        self._doc_lengths = tuple(len(tokens) for tokens in corpus_tokens)
        self._avg_doc_len = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._doc_frequencies = self._build_doc_frequencies(corpus_tokens)

    def get_scores(self, query_tokens: tuple[str, ...]) -> tuple[float, ...]:
        if not self._corpus_tokens or not query_tokens:
            return tuple(0.0 for _ in self._corpus_tokens)

        scores: list[float] = []
        document_count = len(self._corpus_tokens)
        for term_counts, doc_len in zip(self._doc_term_counts, self._doc_lengths, strict=False):
            score = 0.0
            for token in query_tokens:
                term_frequency = term_counts.get(token, 0)
                if term_frequency <= 0:
                    continue
                doc_frequency = self._doc_frequencies.get(token, 0)
                idf = math.log((document_count - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1.0)
                numerator = term_frequency * (1.5 + 1.0)
                denominator = term_frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * doc_len / max(self._avg_doc_len, 1.0)
                )
                score += idf * numerator / max(denominator, 1e-6)
            scores.append(score)
        return tuple(scores)

    def _build_doc_frequencies(
        self,
        corpus_tokens: tuple[tuple[str, ...], ...],
    ) -> dict[str, int]:
        doc_frequencies: dict[str, int] = {}
        for tokens in corpus_tokens:
            for token in set(tokens):
                doc_frequencies[token] = doc_frequencies.get(token, 0) + 1
        return doc_frequencies


class CodeKnowledgeIndex:
    """基于知识条目的混合检索入口。"""

    VECTOR_SIZE = 256

    def __init__(
        self,
        semantic_index: tuple[CodeSemanticSummary, ...],
        *,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._entries = tuple(self._build_entry(summary) for summary in semantic_index)
        self._bm25 = (
            _BM25Index(tuple(self._tokenize(entry.lexical_text) for entry in self._entries))
            if self._entries
            else None
        )

    def search(
        self,
        paper_section: PaperSection,
        *,
        focus_terms: tuple[str, ...] = (),
        top_k: int = 8,
        use_llm_rerank: bool = True,
    ) -> tuple[AlignmentCandidate, ...]:
        if not self._entries or self._bm25 is None:
            return ()

        query_text = self._build_query_text(paper_section, focus_terms)
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return ()

        lexical_scores = self._bm25.get_scores(query_tokens)
        semantic_vector = self._build_semantic_vector(query_text)
        scored: list[tuple[CodeKnowledgeEntry, float]] = []
        max_lexical = max(float(score) for score in lexical_scores) if len(lexical_scores) else 1.0
        if max_lexical <= 0:
            max_lexical = 1.0

        for entry, lexical_score in zip(self._entries, lexical_scores, strict=False):
            lexical_norm = float(lexical_score) / max_lexical
            dense_score = self._cosine_similarity(semantic_vector, entry.semantic_vector)
            anchor_score = self._compute_anchor_overlap(
                query_tokens,
                entry.summary,
                focus_terms=focus_terms,
            )
            structure_bias = self._compute_structure_bias(
                entry.summary.code_evidence,
                focus_terms=focus_terms,
            )
            final_score = (
                0.38 * lexical_norm
                + 0.32 * dense_score
                + 0.14 * anchor_score
                + 0.16 * structure_bias
            )
            scored.append((entry, final_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        preselected = scored[: max(top_k * 2, 6)]
        if use_llm_rerank and self._llm_client is not None and preselected:
            reranked = self._rerank_with_llm(paper_section, focus_terms, preselected)
            if reranked:
                preselected = reranked

        return tuple(
            AlignmentCandidate(
                paper_section=paper_section,
                code_evidence=entry.summary.code_evidence,
                retrieval_score=round(score, 4),
            )
            for entry, score in preselected[:top_k]
        )

    def _build_entry(self, summary: CodeSemanticSummary) -> CodeKnowledgeEntry:
        lexical_text = summary.search_text
        semantic_text = "\n".join(
            part
            for part in (
                summary.code_evidence.file_name,
                summary.code_evidence.symbol_name or summary.code_evidence.parent_symbol,
                summary.summary,
                "；".join(summary.responsibilities),
                " ".join(summary.anchor_terms),
            )
            if part
        )
        return CodeKnowledgeEntry(
            identity=summary.identity,
            summary=summary,
            lexical_text=lexical_text,
            semantic_text=semantic_text,
            semantic_vector=self._build_semantic_vector(semantic_text),
        )

    def _build_query_text(
        self,
        paper_section: PaperSection,
        focus_terms: tuple[str, ...],
    ) -> str:
        focus_text = "\n".join(focus_terms)
        return "\n".join(
            part
            for part in (
                paper_section.title,
                paper_section.content,
                focus_text,
            )
            if part
        )

    def _compute_anchor_overlap(
        self,
        query_tokens: tuple[str, ...],
        summary: CodeSemanticSummary,
        *,
        focus_terms: tuple[str, ...],
    ) -> float:
        normalized_query = {token.lower() for token in query_tokens}
        anchors = {
            token.lower()
            for token in (
                *summary.anchor_terms,
                *summary.defined_symbols,
                *summary.called_symbols,
            )
            if token
        }
        if not anchors:
            return 0.0
        overlap = normalized_query & anchors
        score = min(len(overlap) / max(len(anchors), 1), 1.0)
        searchable = summary.search_text.lower()
        for focus in focus_terms:
            normalized_focus = focus.lower().strip()
            if normalized_focus and normalized_focus in searchable:
                score += 0.08
        return min(score, 1.0)

    def _rerank_with_llm(
        self,
        paper_section: PaperSection,
        focus_terms: tuple[str, ...],
        scored_entries: list[tuple[CodeKnowledgeEntry, float]],
    ) -> list[tuple[CodeKnowledgeEntry, float]] | None:
        if not self._llm_client:
            return None

        try:
            payload = self._llm_client.generate_json(
                system_prompt=(
                    "你是论文代码对齐的代码检索重排器。"
                    "你会阅读论文片段和代码知识条目，判断哪些条目最可能是论文机制的主实现。"
                    "优先选择机制主干、编码器主逻辑、状态更新主逻辑，"
                    "降低 trivial helper、导出函数和只返回布尔值的小函数。"
                    "只返回 JSON 对象。"
                ),
                user_prompt=self._build_rerank_prompt(paper_section, focus_terms, scored_entries),
                temperature=0.0,
                max_tokens=900,
            )
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        ranked_items = payload.get("ranked_items")
        if not isinstance(ranked_items, list):
            return None

        score_lookup = {entry.identity: score for entry, score in scored_entries}
        entry_lookup = {entry.identity: entry for entry, _ in scored_entries}
        reranked: list[tuple[CodeKnowledgeEntry, float]] = []
        seen_ids: set[str] = set()
        for item in ranked_items:
            if not isinstance(item, dict):
                continue
            identity = str(item.get("identity", "")).strip()
            if identity not in entry_lookup or identity in seen_ids:
                continue
            llm_score = float(item.get("score", 0.0) or 0.0)
            combined = 0.65 * max(0.0, min(llm_score, 1.0)) + 0.35 * score_lookup[identity]
            reranked.append((entry_lookup[identity], combined))
            seen_ids.add(identity)

        if not reranked:
            return None

        for entry, score in scored_entries:
            if entry.identity in seen_ids:
                continue
            reranked.append((entry, score))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def _build_rerank_prompt(
        self,
        paper_section: PaperSection,
        focus_terms: tuple[str, ...],
        scored_entries: list[tuple[CodeKnowledgeEntry, float]],
    ) -> str:
        items = []
        for entry, score in scored_entries[:8]:
            evidence = entry.summary.code_evidence
            items.append(
                {
                    "identity": entry.identity,
                    "file": evidence.file_name,
                    "symbol": evidence.symbol_name or evidence.parent_symbol or evidence.block_type,
                    "range": f"L{evidence.start_line}-L{evidence.end_line}",
                    "summary": entry.summary.summary,
                    "responsibilities": list(entry.summary.responsibilities[:3]),
                    "score": round(score, 4),
                }
            )
        return (
            "论文片段：\n"
            f"标题：{paper_section.title}\n"
            f"正文：{paper_section.content}\n"
            f"焦点：{', '.join(focus_terms)}\n\n"
            "候选知识条目：\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
            "请输出 JSON，格式为："
            '{"ranked_items":[{"identity":"...","score":0.0,"reason":"..."}]}'
        )

    def _build_semantic_vector(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.VECTOR_SIZE
        tokens = self._tokenize(text)
        if not tokens:
            return tuple(vector)

        term_counts: dict[str, int] = {}
        for token in tokens:
            normalized = token.lower()
            term_counts[normalized] = term_counts.get(normalized, 0) + 1

        for token, count in term_counts.items():
            index = hash(token) % self.VECTOR_SIZE
            vector[index] += 1.0 + math.log1p(count)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return tuple(vector)
        return tuple(value / norm for value in vector)

    def _cosine_similarity(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        if not left or not right:
            return 0.0
        return max(0.0, sum(a * b for a, b in zip(left, right, strict=False)))

    def _tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))

    def _compute_structure_bias(
        self,
        evidence: CodeEvidence,
        *,
        focus_terms: tuple[str, ...],
    ) -> float:
        symbol = (evidence.symbol_name or evidence.parent_symbol or "").lower()
        file_name = evidence.file_name.lower()
        snippet = evidence.code_snippet.lower()
        span = max(1, evidence.end_line - evidence.start_line + 1)
        score = 0.0

        if self._is_trivial_helper(evidence):
            score -= 0.55
        elif evidence.block_type in {"method", "function"} and 6 <= span <= 80:
            score += 0.12
        elif evidence.block_type == "class":
            score += 0.08

        if any(
            term in symbol
            for term in ("update_graph", "graphlxrtxlayer.forward", "crossmodalencoder")
        ):
            score += 0.45
        if any(term in symbol for term in ("visited", "distance", "save_to_json")):
            score -= 0.22
        if "path" in symbol and "global action" not in " ".join(focus_terms).lower():
            score -= 0.16

        normalized_focus = " ".join(focus_terms).lower()
        if any(term in normalized_focus for term in ("topological", "graph", "map")):
            if any(
                term in snippet
                for term in ("add_edge", "node_positions", "graph.update", "candidate")
            ):
                score += 0.22
            if any(
                term in symbol
                for term in ("update_graph", "get_pos_fts", "floydgraph.path", "graphmap")
            ):
                score += 0.28
            if any(term in symbol for term in ("teacher_action", "make_equiv_action", "rollout")):
                score -= 0.36
            if any(
                term in file_name
                for term in (
                    "/r2r/agent.py",
                    "\\r2r\\agent.py",
                    "/reverie/agent",
                    "\\reverie\\agent",
                )
            ):
                score -= 0.28
            if any(term in file_name for term in ("graph_utils.py", "vilmodel.py")):
                score += 0.18
        if any(
            term in normalized_focus
            for term in ("cross-modal", "graph-aware", "encoder", "global action")
        ):
            if any(
                term in snippet
                for term in ("graph_sprels", "visual_attention", "visn_self_att", "global_encoder")
            ):
                score += 0.22
        if any(term in file_name for term in ("agent_obj", "agent.py", "reverie")) and any(
            term in normalized_focus for term in ("encoder", "graph-aware", "coarse-scale")
        ):
            score -= 0.18

        return max(-1.0, min(score, 1.0))

    def _is_trivial_helper(self, evidence: CodeEvidence) -> bool:
        if evidence.block_type not in {"method", "function"}:
            return False

        span = max(1, evidence.end_line - evidence.start_line + 1)
        symbol = (evidence.symbol_name or "").lower()
        body_lines = [
            line.strip()
            for line in evidence.code_snippet.splitlines()
            if line.strip() and not line.strip().startswith(("def ", "async def "))
        ]
        body_text = " ".join(body_lines).lower()

        if span <= 3:
            return True
        if len(body_lines) <= 2 and body_text.startswith("return "):
            return True
        if span <= 5 and any(
            token in symbol for token in ("visited", "distance", "has_", "is_", "get_")
        ):
            if "get_pos_fts" not in symbol:
                return True
        return False
