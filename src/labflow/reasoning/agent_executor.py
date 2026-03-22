from __future__ import annotations

from dataclasses import replace

from labflow.clients.llm_client import LLMClient
from labflow.reasoning.agent_engine import AgentEventHandler
from labflow.reasoning.agent_engine import (
    PlanAndExecuteEngine as RuntimeEngine,
)
from labflow.reasoning.agent_engine import (
    PlanAndExecuteExecutor as RuntimeExecutor,
)
from labflow.reasoning.agent_engine import (
    PlanAndExecutePlanner as RuntimePlanner,
)
from labflow.reasoning.agent_engine import (
    PlanAndExecuteRePlanner as RuntimeRePlanner,
)
from labflow.reasoning.agent_tools import ReasoningToolbox
from labflow.reasoning.code_grounding_agent import CodeGroundingAgent
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.learning_agents import ReadingAgent, TranslationAgent
from labflow.reasoning.models import (
    AlignmentCandidate,
    AlignmentResult,
    CodeEvidence,
    PaperSection,
)


class PlanAndExecutePlanner(RuntimePlanner):
    """兼容导出：真实实现已经迁移到 agent_engine。"""

    pass


class PlanAndExecuteExecutor(RuntimeExecutor):
    """兼容导出：真实实现已经迁移到 agent_engine。"""

    def __init__(
        self,
        llm_client: LLMClient,
        evidence_builder: EvidenceBuilder,
        tool_registry=None,
    ) -> None:
        super().__init__(
            llm_client,
            evidence_builder,
            tool_registry=tool_registry or ReasoningToolbox(evidence_builder).build_registry(),
        )

    def list_project_structure(self, project_structure: str) -> str:
        return project_structure or "当前代码目录为空。"

    def read_code_segment(
        self,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        path: str,
        line_start: int,
        line_end: int,
    ) -> tuple[str, tuple[str, ...]]:
        return self._evidence_builder.read_logic_block(
            code_evidences,
            path=path,
            line_start=line_start,
            line_end=line_end,
        )

    def llm_semantic_search(
        self,
        *,
        query: str,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        top_k: int = 4,
    ) -> tuple[str, tuple[AlignmentCandidate, ...]]:
        synthetic_section = PaperSection(
            title=paper_section.title,
            content=query,
            level=paper_section.level,
            page_number=paper_section.page_number,
            order=paper_section.order,
        )
        candidates = self._evidence_builder.build_alignment_candidates_from_inputs(
            paper_sections=(synthetic_section,),
            code_evidences=code_evidences,
            top_k=max(2, top_k),
        )
        if not candidates:
            return "语义搜索没有找到相关代码。", ()

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
            )[:top_k]
        )
        lines = [
            f"{candidate.code_evidence.file_name} | "
            f"L{candidate.code_evidence.start_line}-L{candidate.code_evidence.end_line} | "
            f"召回分 {candidate.retrieval_score:.3f}"
            for candidate in ordered
        ]
        return "\n".join(lines), ordered

    def find_definition(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        symbol: str,
        file_path: str,
        line: int,
        column: int,
    ) -> tuple[str, tuple[AlignmentCandidate, ...]]:
        return self._evidence_builder.find_definition_candidate(
            paper_section,
            code_evidences,
            symbol=symbol,
            file_path=file_path,
            line=line,
            column=column,
        )

    def _candidate_id(self, candidate: AlignmentCandidate) -> str:
        evidence = candidate.code_evidence
        return f"{evidence.file_name}:{evidence.start_line}-{evidence.end_line}"


class PlanAndExecuteRePlanner(RuntimeRePlanner):
    """兼容导出：真实实现已经迁移到 agent_engine。"""

    pass


class PlanAndExecuteAgent:
    """总控 Planner 只负责决定启用哪些子 Agent，再把执行交给对应模块。"""

    def __init__(self, llm_client=None, evidence_builder: EvidenceBuilder | None = None) -> None:
        self._llm_client = llm_client or LLMClient()
        self._evidence_builder = evidence_builder or EvidenceBuilder()
        self._tool_registry = ReasoningToolbox(self._evidence_builder).build_registry()
        self.planner = RuntimePlanner(self._llm_client)
        self.executor = RuntimeExecutor(
            self._llm_client,
            self._evidence_builder,
            tool_registry=self._tool_registry,
        )
        self.replanner = RuntimeRePlanner(self._llm_client)
        self.engine = RuntimeEngine(
            planner=self.planner,
            executor=self.executor,
            replanner=self.replanner,
        )
        self.translation_agent = TranslationAgent(self._llm_client)
        self.reading_agent = ReadingAgent(
            self._llm_client,
            translation_agent=self.translation_agent,
        )
        self.code_grounding_agent = CodeGroundingAgent(
            self._llm_client,
            self._evidence_builder,
            self.engine,
        )

    def run(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        project_structure: str,
        event_handler: AgentEventHandler | None = None,
    ) -> AlignmentResult | None:
        section_type, default_agents, code_focus = self._prescreen_section(paper_section)
        role_prompt = self._build_role_prompt(paper_section, section_type)
        plan = self.planner.create_plan(
            paper_section,
            project_structure=project_structure,
            role_prompt=role_prompt,
            default_section_type=section_type,
            default_enabled_agents=default_agents,
            default_code_focus=code_focus,
        )
        enabled_agents = plan.enabled_agents or default_agents
        plan = replace(
            plan,
            enabled_agents=enabled_agents,
            section_type=plan.section_type or section_type,
            code_focus=plan.code_focus or code_focus,
        )
        self._emit(
            event_handler,
            kind="plan_update",
            message=(
                f"Planner 判定当前片段属于 {plan.section_type}，"
                f"启用模块：{', '.join(plan.enabled_agents)}"
            ),
            remaining_steps=tuple(step.display_text for step in plan.steps),
        )

        translation = self._run_translation_agent(
            paper_section,
            enabled_agents=enabled_agents,
            event_handler=event_handler,
        )
        learning_result = self._build_learning_result(
            paper_section,
            translation=translation,
            enabled_agents=enabled_agents,
            event_handler=event_handler,
            planner_note=(
                f"Planner 判定当前片段属于 {plan.section_type}，"
                f"启用模块：{', '.join(plan.enabled_agents)}。"
            ),
        )
        if "code_grounding" not in enabled_agents or not code_evidences:
            if "code_grounding" not in enabled_agents:
                return replace(
                    learning_result,
                    retrieval_plan=(
                        f"{learning_result.retrieval_plan} 当前走学术导读模式，不启动代码对齐。"
                    ),
                    confidence_note="当前片段属于学术导读模式，我不会对它强行发起代码对齐。",
                )
            return learning_result

        self._emit(
            event_handler,
            kind="action",
            message="code_grounding",
            action_input={
                "agent": "code_grounding",
                "focus": list(plan.code_focus[:3]),
            },
        )
        self._emit(
            event_handler,
            kind="thought",
            message="我现在开始把论文机制和候选源码逻辑对起来，必要时会追踪定义源头。",
        )
        code_result = self.code_grounding_agent.run(
            paper_section=paper_section,
            code_evidences=code_evidences,
            project_structure=project_structure,
            role_prompt=role_prompt,
            plan=plan,
            event_handler=event_handler,
        )
        if code_result is None:
            self._emit(
                event_handler,
                kind="observation",
                message="这段内容暂时没有拿到足够可信的源码支撑，我先保留论文导读结果。",
            )
            return replace(
                learning_result,
                confidence_note=(
                    "该段落主要描述理论动机或数学推导，在当前代码库中无直接对应的算子实现，"
                    "建议关注其上游的逻辑设计。"
                ),
            )

        self._emit(
            event_handler,
            kind="observation",
            message="源码对齐阶段完成，已经拿到可展示的实现解释。",
        )
        return self._merge_learning_and_code_result(learning_result, code_result)

    def _run_translation_agent(
        self,
        paper_section: PaperSection,
        *,
        enabled_agents: tuple[str, ...],
        event_handler: AgentEventHandler | None,
    ) -> str:
        if "translation" not in enabled_agents:
            if self._contains_chinese_text(paper_section.content):
                return paper_section.content.strip()
            return self.translation_agent.translate(paper_section)

        self._emit(
            event_handler,
            kind="action",
            message="translate_section",
            action_input={"agent": "translation"},
        )
        self._emit(
            event_handler,
            kind="thought",
            message="我先把当前片段完整翻成中文，后面的要点和术语都建立在这份译文上。",
        )
        return self.translation_agent.translate(paper_section)

    def _build_learning_result(
        self,
        paper_section: PaperSection,
        *,
        translation: str,
        enabled_agents: tuple[str, ...],
        event_handler: AgentEventHandler | None,
        planner_note: str,
    ) -> AlignmentResult:
        if "reading_summary" in enabled_agents or "glossary" in enabled_agents:
            outputs = self.reading_agent.run(
                paper_section,
                translation=translation,
                event_handler=event_handler,
            )
        else:
            outputs = self.reading_agent.run(
                paper_section,
                translation=translation,
                event_handler=None,
            )

        return AlignmentResult(
            paper_section_title=paper_section.title,
            code_file_name="未定位到本地实现",
            alignment_score=0.0,
            match_type="missing_implementation",
            analysis=self.translation_agent.reuse_or_translate(
                paper_section,
                self.translation_agent.normalize_translation_text(outputs.translation),
            ),
            improvement_suggestion="",
            retrieval_score=0.0,
            semantic_evidence=self.reading_agent.normalize_core_points_text(
                outputs.core_points,
                paper_section,
            ),
            research_supplement=self.reading_agent.normalize_glossary_text(
                outputs.glossary,
                paper_section,
            ),
            highlighted_line_numbers=(),
            code_snippet="# 当前未定位到对应源码\n",
            code_start_line=1,
            code_end_line=1,
            retrieval_plan=planner_note,
            implementation_chain="",
            reflection="",
            confidence_note="",
            needs_manual_review=False,
        )

    def _merge_learning_and_code_result(
        self,
        learning_result: AlignmentResult,
        code_result: AlignmentResult,
    ) -> AlignmentResult:
        return replace(
            learning_result,
            code_file_name=code_result.code_file_name,
            alignment_score=code_result.alignment_score,
            match_type=code_result.match_type,
            improvement_suggestion=code_result.improvement_suggestion,
            retrieval_score=code_result.retrieval_score,
            evidence_level=code_result.evidence_level,
            operator_alignment=code_result.operator_alignment,
            shape_alignment=code_result.shape_alignment,
            highlighted_line_numbers=code_result.highlighted_line_numbers,
            code_snippet=code_result.code_snippet,
            code_language=code_result.code_language,
            code_start_line=code_result.code_start_line,
            code_end_line=code_result.code_end_line,
            retrieval_plan=code_result.retrieval_plan,
            implementation_chain=code_result.implementation_chain,
            reflection=code_result.reflection,
            confidence_note=code_result.confidence_note,
            agent_observations=code_result.agent_observations,
            needs_manual_review=code_result.needs_manual_review,
            plan_steps=code_result.plan_steps,
            step_traces=code_result.step_traces,
        )

    def _prescreen_section(
        self,
        paper_section: PaperSection,
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        title = paper_section.title.lower()
        content = paper_section.content.lower()
        combined = f"{title}\n{content}"
        is_chinese = self._contains_chinese_text(paper_section.content)

        if self._looks_like_metadata_section(title, content):
            enabled_agents = ("reading_summary", "glossary")
            if not is_chinese:
                enabled_agents = ("translation",) + enabled_agents
            return "metadata", enabled_agents, ()

        if any(
            keyword in title
            for keyword in ("method", "implementation", "architecture", "approach", "training")
        ) or any(
            keyword in combined
            for keyword in (
                "equation",
                "formula",
                "attention",
                "fusion",
                "loss",
                "decoder",
                "encoder",
                "graph",
                "policy",
                "map",
            )
        ):
            enabled_agents = ("reading_summary", "glossary", "code_grounding")
            if not is_chinese:
                enabled_agents = ("translation",) + enabled_agents
            return "method", enabled_agents, self._infer_code_focus(paper_section)

        if any(
            keyword in title
            for keyword in (
                "abstract",
                "introduction",
                "conclusion",
                "related work",
                "background",
                "motivation",
                "summary",
            )
        ):
            enabled_agents = ("reading_summary", "glossary")
            if not is_chinese:
                enabled_agents = ("translation",) + enabled_agents
            return "academic_guide", enabled_agents, ()

        enabled_agents = ("reading_summary", "glossary", "code_grounding")
        if not is_chinese:
            enabled_agents = ("translation",) + enabled_agents
        return "mixed", enabled_agents, self._infer_code_focus(paper_section)

    def _infer_code_focus(self, paper_section: PaperSection) -> tuple[str, ...]:
        text = paper_section.combined_text
        focus_terms: list[str] = []
        patterns = (
            "global action planning",
            "global planning",
            "coarse-scale",
            "fine-scale",
            "graph-aware",
            "topological map",
            "graph transformer",
            "graph",
            "map",
            "attention",
            "encoder",
            "decoder",
            "fusion",
            "loss",
            "policy",
            "navigation",
            "grounding",
        )
        lowered = text.lower()
        for pattern in patterns:
            if pattern in lowered and pattern not in focus_terms:
                focus_terms.append(pattern)
        if "dual-scale" in lowered:
            focus_terms.append("dual-scale local global encoder")
        return tuple(focus_terms[:4])

    def _looks_like_metadata_section(self, title: str, content: str) -> bool:
        if "arxiv:" in title or "arxiv:" in content:
            return True
        metadata_terms = ("university", "institute", "author", "correspondence", "@")
        return any(term in content for term in metadata_terms)

    def _contains_chinese_text(self, text: str) -> bool:
        return sum("\u4e00" <= char <= "\u9fff" for char in text) >= 8

    def _build_role_prompt(self, paper_section: PaperSection, section_type: str) -> str:
        title = paper_section.title.lower()
        if "introduction" in title:
            role = "你现在的角色是科研领路人，请解释这段话的研究背景、问题定义和核心创新点。"
        elif section_type in {"method", "mixed"}:
            role = (
                "你现在的角色是代码审计专家，请寻找这段公式或机制的具体代码落地，"
                "宁可拒绝硬凑，也不要给出幻觉式映射。"
            )
        else:
            role = "你现在的角色是科研领路人，请提炼核心思想、理论动机和阅读重点。"
        return "\n".join([self._llm_client.get_react_agent_role_prompt(), role])

    def _emit(self, handler: AgentEventHandler | None, **payload: object) -> None:
        if handler is not None:
            handler(payload)
