"""推理层数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MatchType = Literal[
    "strong_match",
    "partial_match",
    "missing_implementation",
    "formula_mismatch",
]


@dataclass(frozen=True)
class PaperSection:
    """论文中的章节片段。"""

    title: str
    content: str
    level: int
    page_number: int
    order: int

    @property
    def combined_text(self) -> str:
        """拼接章节标题与正文，供后续检索与提示词使用。"""

        return f"{self.title}\n{self.content}".strip()


@dataclass(frozen=True)
class CodeEvidence:
    """代码侧证据。"""

    file_name: str
    code_snippet: str
    related_git_diff: str
    symbols: tuple[str, ...]
    commit_context: tuple[str, ...]

    @property
    def combined_text(self) -> str:
        """把代码片段、diff 和提交上下文揉成统一检索语料。"""

        commit_text = "\n".join(self.commit_context)
        return "\n".join(
            part
            for part in (self.file_name, self.code_snippet, self.related_git_diff, commit_text)
            if part
        )


@dataclass(frozen=True)
class AlignmentCandidate:
    """候选对齐对。"""

    paper_section: PaperSection
    code_evidence: CodeEvidence
    retrieval_score: float


@dataclass(frozen=True)
class AlignmentResult:
    """结构化对齐分析结果。"""

    paper_section_title: str
    code_file_name: str
    alignment_score: float
    match_type: MatchType
    analysis: str
    improvement_suggestion: str
    retrieval_score: float

    @classmethod
    def from_payload(cls, payload: dict, candidate: AlignmentCandidate) -> AlignmentResult:
        """把模型返回的 JSON 整理成稳定结构。"""

        raw_score = float(
            payload.get("alignment_score", payload.get("score", candidate.retrieval_score))
        )
        normalized_score = max(0.0, min(1.0, raw_score))
        raw_match_type = str(payload.get("match_type", "partial_match"))
        if raw_match_type not in {
            "strong_match",
            "partial_match",
            "missing_implementation",
            "formula_mismatch",
        }:
            raw_match_type = "partial_match"

        return cls(
            paper_section_title=candidate.paper_section.title,
            code_file_name=candidate.code_evidence.file_name,
            alignment_score=normalized_score,
            match_type=raw_match_type,
            analysis=str(payload.get("analysis", "当前结论为空，需要补充模型分析。")).strip(),
            improvement_suggestion=str(
                payload.get("improvement_suggestion", "先补齐证据，再继续对齐。")
            ).strip(),
            retrieval_score=candidate.retrieval_score,
        )
