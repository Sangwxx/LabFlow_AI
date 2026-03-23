"""推理层工具注册表。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from labflow.clients.llm_client import LLMClient
from labflow.reasoning.code_knowledge_index import CodeKnowledgeIndex
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import AlignmentCandidate, CodeEvidence, PaperSection, ToolName


@dataclass(frozen=True)
class AgentToolContext:
    """工具执行时共享的上下文。"""

    paper_section: PaperSection
    project_structure: str
    code_evidences: tuple[CodeEvidence, ...]
    current_candidates: tuple[AlignmentCandidate, ...]


@dataclass(frozen=True)
class ToolExecutionResult:
    """单次工具调用的稳定输出。"""

    tool_name: ToolName
    tool_input: str
    observation: str
    candidates: tuple[AlignmentCandidate, ...]


@dataclass(frozen=True)
class AgentTool:
    """注册表中的工具定义。"""

    name: ToolName
    handler: Callable[[object, AgentToolContext], ToolExecutionResult]


class ToolRegistry:
    """统一管理工具注册与执行。"""

    def __init__(self, tools: tuple[AgentTool, ...]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def execute(
        self,
        name: str,
        action_input: object,
        context: AgentToolContext,
    ) -> ToolExecutionResult:
        tool = self._tools.get(name)
        if tool is None:
            tool = self._tools["llm_semantic_search"]
        return tool.handler(action_input, context)

    @property
    def tool_names(self) -> tuple[ToolName, ...]:
        return tuple(self._tools.keys())


class ReasoningToolbox:
    """默认工具箱实现。"""

    def __init__(
        self,
        evidence_builder: EvidenceBuilder,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._evidence_builder = evidence_builder
        self._llm_client = llm_client

    def build_registry(self) -> ToolRegistry:
        return ToolRegistry(
            (
                AgentTool("list_project_structure", self._handle_list_project_structure),
                AgentTool("read_code_segment", self._handle_read_code_segment),
                AgentTool("llm_semantic_search", self._handle_llm_semantic_search),
                AgentTool("find_definition", self._handle_find_definition),
            )
        )

    def _handle_list_project_structure(
        self,
        action_input: object,
        context: AgentToolContext,
    ) -> ToolExecutionResult:
        _ = action_input
        return ToolExecutionResult(
            tool_name="list_project_structure",
            tool_input="查看项目结构",
            observation=context.project_structure or "当前代码目录为空。",
            candidates=context.current_candidates,
        )

    def _handle_read_code_segment(
        self,
        action_input: object,
        context: AgentToolContext,
    ) -> ToolExecutionResult:
        params = action_input if isinstance(action_input, dict) else {}
        path = str(params.get("path") or params.get("file_path") or "").strip()
        line_start = int(params.get("line_start", params.get("start_line", 1)))
        line_end = int(params.get("line_end", params.get("end_line", line_start)))
        observation, candidate_ids = self._evidence_builder.read_logic_block(
            context.code_evidences,
            path=path,
            line_start=line_start,
            line_end=line_end,
        )
        selected = [
            candidate
            for candidate in context.current_candidates
            if self._candidate_id(candidate) in set(candidate_ids)
        ]
        return ToolExecutionResult(
            tool_name="read_code_segment",
            tool_input=f"{path}:{line_start}-{line_end}",
            observation=observation,
            candidates=tuple(selected or context.current_candidates),
        )

    def _handle_llm_semantic_search(
        self,
        action_input: object,
        context: AgentToolContext,
    ) -> ToolExecutionResult:
        params = action_input if isinstance(action_input, dict) else {}
        query = str(params.get("query", "")).strip() or context.paper_section.combined_text
        semantic_index = self._evidence_builder.build_semantic_index_from_evidences(
            context.code_evidences,
            llm_client=self._llm_client,
        )
        synthetic_section = PaperSection(
            title=context.paper_section.title,
            content=query,
            level=context.paper_section.level,
            page_number=context.paper_section.page_number,
            order=context.paper_section.order,
        )
        knowledge_index = CodeKnowledgeIndex(
            semantic_index,
            llm_client=self._llm_client,
        )
        candidates = knowledge_index.search(
            synthetic_section,
            focus_terms=tuple(query.split()),
            top_k=6,
            use_llm_rerank=False,
        )
        traced_candidates = self._evidence_builder.trace_related_candidates(
            synthetic_section,
            semantic_index,
            trace_symbols=tuple(query.split()),
            seen_candidate_ids={self._candidate_id(item) for item in candidates},
            limit=2,
        )
        if traced_candidates:
            candidates = tuple((*candidates, *traced_candidates))
        if not candidates:
            return ToolExecutionResult(
                tool_name="llm_semantic_search",
                tool_input=query,
                observation="语义搜索没有找到相关代码。",
                candidates=context.current_candidates,
            )

        dedup: dict[str, AlignmentCandidate] = {}
        for candidate in candidates:
            candidate_id = self._candidate_id(candidate)
            if (
                candidate_id not in dedup
                or candidate.retrieval_score > dedup[candidate_id].retrieval_score
            ):
                dedup[candidate_id] = candidate

        ordered = tuple(
            sorted(
                dedup.values(),
                key=lambda item: item.retrieval_score,
                reverse=True,
            )[:4]
        )
        lines = [self._format_candidate_summary(candidate) for candidate in ordered]
        return ToolExecutionResult(
            tool_name="llm_semantic_search",
            tool_input=query,
            observation="\n".join(lines),
            candidates=ordered,
        )

    def _handle_find_definition(
        self,
        action_input: object,
        context: AgentToolContext,
    ) -> ToolExecutionResult:
        params = action_input if isinstance(action_input, dict) else {}
        symbol = str(params.get("symbol", "")).strip()
        file_path = str(params.get("file_path", "")).strip()
        line = int(params.get("line", 1))
        column = int(params.get("column", 0))
        observation, candidates = self._evidence_builder.find_definition_candidate(
            context.paper_section,
            context.code_evidences,
            symbol=symbol,
            file_path=file_path,
            line=line,
            column=column,
        )
        return ToolExecutionResult(
            tool_name="find_definition",
            tool_input=f"{symbol} @ {file_path}:{line}:{column}",
            observation=observation,
            candidates=candidates or context.current_candidates,
        )

    def _candidate_id(self, candidate: AlignmentCandidate) -> str:
        evidence = candidate.code_evidence
        return f"{evidence.file_name}:{evidence.start_line}-{evidence.end_line}"

    def _format_candidate_summary(self, candidate: AlignmentCandidate) -> str:
        evidence = candidate.code_evidence
        symbol = evidence.symbol_name or "未命名逻辑块"
        return (
            f"{evidence.file_name} | "
            f"L{evidence.start_line}-L{evidence.end_line} | "
            f"{evidence.block_type} | {symbol} | "
            f"召回分 {candidate.retrieval_score}"
        )
