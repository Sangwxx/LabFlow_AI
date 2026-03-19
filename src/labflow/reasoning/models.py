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

ToolName = Literal[
    "list_project_structure",
    "read_code_segment",
    "llm_semantic_search",
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
    start_line: int = 1
    end_line: int = 1
    language: str = "python"

    @property
    def combined_text(self) -> str:
        """把代码片段、diff 和提交上下文拉成统一检索语料。"""

        commit_text = "\n".join(self.commit_context)
        return "\n".join(
            part
            for part in (self.file_name, self.code_snippet, self.related_git_diff, commit_text)
            if part
        )


@dataclass(frozen=True)
class CodeSemanticSummary:
    """仓库语义索引条目。"""

    code_evidence: CodeEvidence
    summary: str
    responsibilities: tuple[str, ...]
    defined_symbols: tuple[str, ...]
    called_symbols: tuple[str, ...]
    anchor_terms: tuple[str, ...]

    @property
    def search_text(self) -> str:
        """把摘要、职责和原始代码聚合成检索文本。"""

        return "\n".join(
            item
            for item in (
                self.code_evidence.file_name,
                self.summary,
                "\n".join(self.responsibilities),
                " ".join(self.defined_symbols),
                " ".join(self.called_symbols),
                " ".join(self.anchor_terms),
                self.code_evidence.code_snippet,
            )
            if item
        )

    @property
    def identity(self) -> str:
        """给规划执行循环一个稳定的证据标识。"""

        return (
            f"{self.code_evidence.file_name}:"
            f"{self.code_evidence.start_line}-{self.code_evidence.end_line}"
        )


@dataclass(frozen=True)
class AlignmentCandidate:
    """候选对齐对。"""

    paper_section: PaperSection
    code_evidence: CodeEvidence
    retrieval_score: float


@dataclass(frozen=True)
class PlanStep:
    """规划器产出的单个步骤。"""

    step_id: str
    description: str
    objective: str = ""

    @property
    def display_text(self) -> str:
        """整理成 UI 可直接展示的文本。"""

        if self.objective:
            return f"{self.step_id}. {self.description} | 目标: {self.objective}"
        return f"{self.step_id}. {self.description}"


@dataclass(frozen=True)
class ExecutionPlan:
    """Planner / RePlanner 维护的计划状态。"""

    steps: tuple[PlanStep, ...]
    rationale: str = ""
    is_finished: bool = False
    final_summary: str = ""


@dataclass(frozen=True)
class ToolInvocation:
    """Executor 一次工具调用的记录。"""

    tool_name: ToolName
    tool_input: str
    observation: str


@dataclass(frozen=True)
class StepExecutionTrace:
    """单个计划步骤的执行轨迹。"""

    step: PlanStep
    thought: str
    action: str
    observation: str
    tool_invocations: tuple[ToolInvocation, ...]
    produced_candidate_ids: tuple[str, ...] = ()


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
    semantic_evidence: str = ""
    research_supplement: str = ""
    highlighted_line_numbers: tuple[int, ...] = ()
    code_snippet: str = ""
    code_language: str = "python"
    code_start_line: int = 1
    code_end_line: int = 1
    retrieval_plan: str = ""
    implementation_chain: str = ""
    reflection: str = ""
    confidence_note: str = ""
    agent_observations: tuple[str, ...] = ()
    needs_manual_review: bool = False
    plan_steps: tuple[PlanStep, ...] = ()
    step_traces: tuple[StepExecutionTrace, ...] = ()

    @property
    def score_out_of_ten(self) -> float:
        """把 0-1 分值映射为 10 分制。"""

        return round(self.alignment_score * 10, 1)

    @property
    def is_high_risk(self) -> bool:
        """低分项和明显错配项都应优先报警。"""

        return self.alignment_score < 0.6 or self.match_type in {
            "missing_implementation",
            "formula_mismatch",
        }

    @property
    def is_good_alignment(self) -> bool:
        """高置信度一致项单独归类。"""

        return self.alignment_score >= 0.75 and self.match_type == "strong_match"

    @classmethod
    def from_payload(cls, payload: dict, candidate: AlignmentCandidate) -> AlignmentResult:
        """把模型返回的 JSON 整理成稳定结果。"""

        raw_score = float(
            payload.get("alignment_score", payload.get("score", candidate.retrieval_score))
        )
        normalized_score = max(0.0, min(1.0, raw_score))
        raw_match_type = str(payload.get("match_type", "partial_match")).strip()
        if raw_match_type not in {
            "strong_match",
            "partial_match",
            "missing_implementation",
            "formula_mismatch",
        }:
            raw_match_type = "partial_match"

        highlighted_lines = cls._normalize_highlighted_lines(payload, candidate)

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
            semantic_evidence=str(
                payload.get("semantic_evidence", payload.get("analysis", "当前缺少语义证据。"))
            ).strip(),
            research_supplement=str(payload.get("research_supplement", "")).strip(),
            highlighted_line_numbers=highlighted_lines,
            code_snippet=candidate.code_evidence.code_snippet,
            code_language=candidate.code_evidence.language,
            code_start_line=candidate.code_evidence.start_line,
            code_end_line=candidate.code_evidence.end_line,
            retrieval_plan=str(payload.get("retrieval_plan", "")).strip(),
            implementation_chain=str(
                payload.get("implementation_chain", payload.get("analysis", ""))
            ).strip(),
            reflection=str(payload.get("reflection", "")).strip(),
            confidence_note=str(payload.get("confidence_note", "")).strip(),
            needs_manual_review=bool(payload.get("needs_manual_review", False)),
        )

    @staticmethod
    def _normalize_highlighted_lines(
        payload: dict,
        candidate: AlignmentCandidate,
    ) -> tuple[int, ...]:
        """把模型返回的高亮行收敛到当前代码片段范围内。"""

        raw_lines = payload.get("highlighted_lines", [])
        if not isinstance(raw_lines, list):
            raw_lines = []

        normalized_lines: list[int] = []
        for raw_line in raw_lines:
            try:
                line_number = int(raw_line)
            except (TypeError, ValueError):
                continue
            if (
                candidate.code_evidence.start_line
                <= line_number
                <= candidate.code_evidence.end_line
            ):
                normalized_lines.append(line_number)

        if normalized_lines:
            return tuple(dict.fromkeys(normalized_lines))

        if candidate.code_evidence.start_line <= candidate.code_evidence.end_line:
            return (candidate.code_evidence.start_line,)
        return ()
