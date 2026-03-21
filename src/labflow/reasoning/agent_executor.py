from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import replace

from labflow.clients.llm_client import LLMClient
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
from labflow.reasoning.agent_prompts import (
    build_final_answer_system_prompt,
    build_final_answer_user_prompt,
    build_learning_core_points_system_prompt,
    build_learning_glossary_system_prompt,
    build_reflection_system_prompt,
    build_reflection_user_prompt,
    build_translation_json_system_prompt,
    build_translation_segment_system_prompt,
    build_translation_text_system_prompt,
)
from labflow.reasoning.agent_tools import ReasoningToolbox
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import (
    AlignmentCandidate,
    AlignmentResult,
    CodeEvidence,
    ExecutionPlan,
    PaperSection,
    PlanStep,
    StepExecutionTrace,
    ToolInvocation,
)

AgentEventHandler = Callable[[dict], None]


class PlanAndExecutePlanner:
    """先拆计划，再让执行器按需追逻辑。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def create_plan(
        self,
        paper_section: PaperSection,
        *,
        project_structure: str,
        role_prompt: str,
    ) -> ExecutionPlan:
        schema = {
            "rationale": "中文规划思路",
            "steps": [{"description": "中文步骤", "objective": "目标"}],
        }
        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    role_prompt,
                    "你当前扮演 Planner。",
                    "请把任务拆成 2 到 4 个可执行步骤。",
                    "优先定位实现入口、追踪定义、核对算子与张量形状。",
                    "不要把重点放在变量名映射上。",
                    f"只输出 JSON，格式为: {json.dumps(schema, ensure_ascii=False)}",
                ]
            ),
            user_prompt=(
                f"【论文片段】{paper_section.combined_text}\n\n"
                f"【项目文件树】\n{project_structure or '当前未提供文件树。'}"
            ),
            temperature=0.1,
            max_tokens=1000,
        )
        if not isinstance(payload, dict):
            return self._fallback_plan()

        steps = self._normalize_steps(payload.get("steps"))
        return ExecutionPlan(
            steps=steps or self._fallback_plan().steps,
            rationale=(
                str(payload.get("rationale", "")).strip() or self._fallback_plan().rationale
            ),
        )

    def _normalize_steps(self, raw_steps: object) -> tuple[PlanStep, ...]:
        steps: list[PlanStep] = []
        if not isinstance(raw_steps, list):
            return ()
        for index, item in enumerate(raw_steps, start=1):
            if isinstance(item, dict):
                description = str(item.get("description", "")).strip()
                objective = str(item.get("objective", "")).strip()
            else:
                description = str(item).strip()
                objective = ""
            if description:
                steps.append(PlanStep(str(index), description, objective))
        return tuple(steps)

    def _fallback_plan(self) -> ExecutionPlan:
        return ExecutionPlan(
            steps=(
                PlanStep("1", "扫描文件树并锁定最可能的实现入口", "缩小目标范围"),
                PlanStep("2", "读取完整逻辑块并在必要时追踪定义源头", "形成实现链"),
                PlanStep("3", "核对算子、输入输出维度与论文步骤", "生成高可信结论"),
                PlanStep("4", "若本地缺源码则切换到学术解释模式", "保证不断流"),
            ),
            rationale="先定入口，再追定义，最后核对算子和形状；没有源码就讲清论文。",
        )


class PlanAndExecuteExecutor:
    """执行计划步骤，并把工具调用暴露给 UI。"""

    def __init__(self, llm_client: LLMClient, evidence_builder: EvidenceBuilder) -> None:
        self._llm_client = llm_client
        self._evidence_builder = evidence_builder

    def execute(
        self,
        step: PlanStep,
        *,
        paper_section: PaperSection,
        project_structure: str,
        code_evidences: tuple[CodeEvidence, ...],
        current_candidates: tuple[AlignmentCandidate, ...],
        role_prompt: str,
        event_handler: AgentEventHandler | None = None,
        max_actions: int = 3,
    ) -> tuple[StepExecutionTrace, tuple[AlignmentCandidate, ...]]:
        tool_invocations: list[ToolInvocation] = []
        latest_candidates = current_candidates
        thought = ""
        action = ""
        observation = ""

        for _ in range(max_actions):
            payload = self._llm_client.generate_json(
                system_prompt=self._build_executor_system_prompt(role_prompt),
                user_prompt=self._build_executor_user_prompt(
                    step=step,
                    paper_section=paper_section,
                    project_structure=project_structure,
                    current_candidates=latest_candidates,
                    tool_invocations=tuple(tool_invocations),
                ),
                temperature=0.1,
                max_tokens=1000,
            )
            if not isinstance(payload, dict):
                payload = {
                    "thought": "当前模型没有稳定返回结构化动作，我先结束这一步。",
                    "action": "finish",
                    "action_input": {},
                    "final_observation": "当前步骤先用已有证据继续推进。",
                }

            thought = str(payload.get("thought", "继续核查实现链。")).strip()
            self._emit(event_handler, kind="thought", message=thought, step=step.display_text)

            action = str(payload.get("action", "finish")).strip()
            action_input = payload.get("action_input", {})
            self._emit(
                event_handler,
                kind="action",
                message=action,
                step=step.display_text,
                action_input=action_input,
            )

            if action == "finish":
                observation = str(payload.get("final_observation", "当前步骤执行完成。")).strip()
                self._emit(
                    event_handler,
                    kind="observation",
                    message=observation,
                    step=step.display_text,
                )
                break

            tool_result, latest_candidates = self._invoke_tool(
                action=action,
                action_input=action_input,
                paper_section=paper_section,
                project_structure=project_structure,
                code_evidences=code_evidences,
                current_candidates=latest_candidates,
            )
            observation = tool_result["observation"]
            tool_invocations.append(
                ToolInvocation(
                    tool_name=tool_result["tool_name"],
                    tool_input=tool_result["tool_input"],
                    observation=observation,
                )
            )
            self._emit(
                event_handler,
                kind="observation",
                message=observation,
                step=step.display_text,
            )

        return (
            StepExecutionTrace(
                step=step,
                thought=thought,
                action=action,
                observation=observation,
                tool_invocations=tuple(tool_invocations),
                produced_candidate_ids=tuple(
                    self._candidate_id(candidate) for candidate in latest_candidates
                ),
            ),
            latest_candidates,
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
        lines = [self._format_candidate_summary(candidate) for candidate in ordered]
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

    def _invoke_tool(
        self,
        *,
        action: str,
        action_input: object,
        paper_section: PaperSection,
        project_structure: str,
        code_evidences: tuple[CodeEvidence, ...],
        current_candidates: tuple[AlignmentCandidate, ...],
    ) -> tuple[dict, tuple[AlignmentCandidate, ...]]:
        if action == "list_project_structure":
            return (
                {
                    "tool_name": "list_project_structure",
                    "tool_input": "查看项目结构",
                    "observation": self.list_project_structure(project_structure),
                },
                current_candidates,
            )

        if action == "read_code_segment":
            params = action_input if isinstance(action_input, dict) else {}
            path = str(params.get("path", "")).strip()
            line_start = int(params.get("line_start", 1))
            line_end = int(params.get("line_end", line_start))
            observation, candidate_ids = self.read_code_segment(
                code_evidences,
                path=path,
                line_start=line_start,
                line_end=line_end,
            )
            selected = [
                candidate
                for candidate in current_candidates
                if self._candidate_id(candidate) in set(candidate_ids)
            ]
            return (
                {
                    "tool_name": "read_code_segment",
                    "tool_input": f"{path}:{line_start}-{line_end}",
                    "observation": observation,
                },
                tuple(selected or current_candidates),
            )

        if action == "find_definition":
            params = action_input if isinstance(action_input, dict) else {}
            symbol = str(params.get("symbol", "")).strip()
            file_path = str(params.get("file_path", "")).strip()
            line = int(params.get("line", 1))
            column = int(params.get("column", 0))
            observation, candidates = self.find_definition(
                paper_section,
                code_evidences,
                symbol=symbol,
                file_path=file_path,
                line=line,
                column=column,
            )
            return (
                {
                    "tool_name": "find_definition",
                    "tool_input": f"{symbol} @ {file_path}:{line}:{column}",
                    "observation": observation,
                },
                candidates or current_candidates,
            )

        params = action_input if isinstance(action_input, dict) else {}
        query = str(params.get("query", "")).strip() or paper_section.combined_text
        observation, searched = self.llm_semantic_search(
            query=query,
            paper_section=paper_section,
            code_evidences=code_evidences,
            top_k=4,
        )
        return (
            {
                "tool_name": "llm_semantic_search",
                "tool_input": query,
                "observation": observation,
            },
            searched or current_candidates,
        )

    def _build_executor_system_prompt(self, role_prompt: str) -> str:
        schema = {
            "thought": "当前步骤的思考",
            "action": (
                "list_project_structure | read_code_segment | "
                "llm_semantic_search | find_definition | finish"
            ),
            "action_input": "对象形式的工具入参",
            "final_observation": "当 action=finish 时的阶段结论",
        }
        return "\n".join(
            [
                role_prompt,
                "你当前扮演 Executor。",
                "你必须先给出 Thought，再选择一个 Action。",
                "如果遇到不确定的类、函数或方法，应主动调用 find_definition 追到定义源头。",
                "不要输出变量映射表，要解释实现逻辑。",
                f"只输出 JSON，格式为: {json.dumps(schema, ensure_ascii=False)}",
            ]
        )

    def _build_executor_user_prompt(
        self,
        *,
        step: PlanStep,
        paper_section: PaperSection,
        project_structure: str,
        current_candidates: tuple[AlignmentCandidate, ...],
        tool_invocations: tuple[ToolInvocation, ...],
    ) -> str:
        candidates = (
            "\n".join(
                self._format_candidate_summary(candidate, index)
                for index, candidate in enumerate(current_candidates[:4])
            )
            or "暂无候选"
        )
        history = (
            "\n".join(
                [
                    f"工具: {item.tool_name}\n输入: {item.tool_input}\n观察: {item.observation}"
                    for item in tool_invocations
                ]
            )
            or "无"
        )
        return (
            f"【步骤】{step.display_text}\n"
            f"【论文片段】{paper_section.combined_text}\n\n"
            f"【项目结构】\n{project_structure}\n\n"
            f"【当前候选】\n{candidates}\n\n"
            f"【已执行历史】\n{history}"
        )

    def _format_candidate_summary(
        self,
        candidate: AlignmentCandidate,
        index: int | None = None,
    ) -> str:
        evidence = candidate.code_evidence
        prefix = f"候选 {index} | " if index is not None else ""
        symbol = evidence.symbol_name or "未命名逻辑块"
        return (
            f"{prefix}{evidence.file_name} | "
            f"L{evidence.start_line}-L{evidence.end_line} | "
            f"{evidence.block_type} | {symbol} | "
            f"召回分 {candidate.retrieval_score}"
        )

    def _candidate_id(self, candidate: AlignmentCandidate) -> str:
        return self._candidate_id_from_evidence(candidate.code_evidence)

    def _candidate_id_from_evidence(self, evidence: CodeEvidence) -> str:
        return f"{evidence.file_name}:{evidence.start_line}-{evidence.end_line}"

    def _emit(self, handler: AgentEventHandler | None, **payload: object) -> None:
        if handler is not None:
            handler(payload)


class PlanAndExecuteRePlanner:
    """根据 Observation 决定是否继续剩余计划。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def update_plan(
        self,
        plan: ExecutionPlan,
        trace: StepExecutionTrace,
        *,
        role_prompt: str,
    ) -> ExecutionPlan:
        remaining_steps = plan.steps[1:]
        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    role_prompt,
                    "你当前扮演 RePlanner。",
                    "请根据刚完成步骤的 Observation 判断是否继续执行剩余计划。",
                    (
                        "只输出 JSON，格式为 "
                        '{"is_finished": true/false, "final_summary": "...", '
                        '"remaining_steps": [...]}。'
                    ),
                ]
            ),
            user_prompt=(
                f"【原计划】\n{chr(10).join(step.display_text for step in plan.steps)}\n\n"
                f"【刚完成步骤】{trace.step.display_text}\n"
                f"【Thought】{trace.thought}\n"
                f"【Action】{trace.action}\n"
                f"【Observation】{trace.observation}\n\n"
                f"【剩余步骤】\n"
                f"{chr(10).join(step.display_text for step in remaining_steps) or '无'}"
            ),
            temperature=0.1,
            max_tokens=700,
        )
        if not isinstance(payload, dict):
            return ExecutionPlan(
                steps=remaining_steps,
                rationale=plan.rationale,
                is_finished=not remaining_steps,
                final_summary="",
            )

        return ExecutionPlan(
            steps=remaining_steps,
            rationale=plan.rationale,
            is_finished=bool(payload.get("is_finished", False)) or not remaining_steps,
            final_summary=str(payload.get("final_summary", "")).strip(),
        )


class PlanAndExecuteAgent:
    """按需触发的 Planner / Executor / RePlanner 控制中心。"""

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

    def run(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        project_structure: str,
        event_handler: AgentEventHandler | None = None,
    ) -> AlignmentResult | None:
        section_mode = self._classify_section_mode(paper_section)
        role_prompt = self._build_role_prompt(paper_section, section_mode)

        if section_mode == "academic_guide":
            self._emit(
                event_handler,
                kind="thought",
                message="当前片段属于标题、摘要、引言或结论类内容，我先切换到学术导读模式，不启动代码对齐。",
            )
            return self._finalize_learning_result(
                self._build_academic_only_result(
                    paper_section,
                    role_prompt=role_prompt,
                    mode_note="当前片段被预审为学术导读内容，未启动代码深度对齐。",
                    event_handler=event_handler,
                ),
                paper_section,
            )

        if not code_evidences:
            return self._finalize_learning_result(
                self._build_graceful_refusal_result(
                    paper_section,
                    reason="当前代码库里还没有可核对的本地源码，因此我先转为论文导读与理论解释。",
                ),
                paper_section,
            )

        current_candidates = self._evidence_builder.build_alignment_candidates_from_inputs(
            paper_sections=(paper_section,),
            code_evidences=code_evidences,
            top_k=4,
        )
        if not current_candidates:
            return self._finalize_learning_result(
                self._build_graceful_refusal_result(
                    paper_section,
                    reason=(
                        "该段落更像理论动机或数学推导，我在当前代码库中没有定位到直接承接它的算子实现。"
                    ),
                ),
                paper_section,
            )

        current_plan, step_traces, current_candidates = self.engine.run(
            paper_section=paper_section,
            project_structure=project_structure,
            code_evidences=code_evidences,
            current_candidates=current_candidates,
            role_prompt=role_prompt,
            event_handler=event_handler,
        )

        result = self._build_final_answer(
            paper_section=paper_section,
            current_candidates=current_candidates,
            step_traces=step_traces,
            current_plan=current_plan,
            role_prompt=role_prompt,
        )
        reflected_result = self._reflect(
            result,
            paper_section=paper_section,
            role_prompt=role_prompt,
        )
        if self._should_refuse_alignment(
            paper_section,
            reflected_result,
            current_candidates,
        ):
            return self._finalize_learning_result(
                self._build_graceful_refusal_result(
                    paper_section,
                    reason=(
                        "该段落主要描述理论动机或数学推导，在当前代码库中无直接对应的算子实现，建议关注其上游的逻辑设计。"
                    ),
                    fallback_candidate=current_candidates[0],
                ),
                paper_section,
            )
        return self._finalize_learning_result(reflected_result, paper_section)

    def _build_final_answer(
        self,
        *,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        step_traces: tuple[StepExecutionTrace, ...],
        current_plan: ExecutionPlan,
        role_prompt: str,
    ) -> AlignmentResult:
        payload = self._llm_client.generate_json(
            system_prompt=build_final_answer_system_prompt(role_prompt),
            user_prompt=build_final_answer_user_prompt(
                paper_section,
                current_candidates,
                step_traces,
            ),
            temperature=0.1,
            max_tokens=2000,
        )
        if not isinstance(payload, dict):
            return self._build_local_fallback_result(
                paper_section,
                current_candidates,
                step_traces,
                current_plan,
            )

        try:
            selected_index = int(payload.get("best_candidate_index", 0))
        except (TypeError, ValueError):
            selected_index = 0
        selected_index = max(0, min(selected_index, len(current_candidates) - 1))

        result = AlignmentResult.from_payload(payload, current_candidates[selected_index])
        plan_steps = current_plan.steps or tuple(trace.step for trace in step_traces)
        evidence = current_candidates[selected_index].code_evidence
        return replace(
            result,
            retrieval_plan="\n".join(step.display_text for step in plan_steps),
            plan_steps=plan_steps,
            step_traces=step_traces,
            agent_observations=tuple(trace.observation for trace in step_traces),
            research_supplement=(
                result.research_supplement or self._build_term_glossary(paper_section)
            ),
            operator_alignment=(
                result.operator_alignment or self._build_operator_alignment(paper_section, evidence)
            ),
            shape_alignment=(
                result.shape_alignment or self._build_shape_alignment(paper_section, evidence)
            ),
            implementation_chain=(
                result.implementation_chain
                if self._has_strong_source_grounding(result, evidence)
                else ""
            ),
        )

    def _reflect(
        self,
        result: AlignmentResult,
        *,
        paper_section: PaperSection,
        role_prompt: str,
    ) -> AlignmentResult:
        payload = self._llm_client.generate_json(
            system_prompt=build_reflection_system_prompt(role_prompt),
            user_prompt=build_reflection_user_prompt(paper_section, result),
            temperature=0.0,
            max_tokens=900,
        )
        if not isinstance(payload, dict):
            payload = {
                "reflection": "当前没有稳定拿到反思结果，建议人工核对。",
                "final_confidence": result.alignment_score,
                "confidence_note": "模型反思阶段未稳定返回，建议人工复核。",
                "needs_manual_review": True,
            }

        try:
            reflected_score = float(payload.get("final_confidence", result.alignment_score))
        except (TypeError, ValueError):
            reflected_score = result.alignment_score
        reflected_score = max(0.0, min(1.0, reflected_score))

        confidence_note = str(payload.get("confidence_note", "")).strip()
        needs_manual_review = bool(payload.get("needs_manual_review", False))
        if reflected_score < 0.8 and not confidence_note:
            confidence_note = "当前证据链还不够扎实，建议你人工复核关键实现行。"

        return replace(
            result,
            alignment_score=reflected_score,
            reflection=str(payload.get("reflection", "当前缺少自我审计结论。")).strip(),
            confidence_note=confidence_note,
            needs_manual_review=needs_manual_review or reflected_score < 0.8,
        )

    def _classify_section_mode(self, paper_section: PaperSection) -> str:
        title = paper_section.title.lower()
        content = paper_section.content.lower()
        combined = f"{title}\n{content}"

        academic_keywords = (
            "abstract",
            "introduction",
            "conclusion",
            "related work",
            "background",
            "motivation",
            "summary",
            "discussion",
            "preliminar",
            "limitation",
        )
        code_keywords = (
            "method",
            "implementation",
            "architecture",
            "approach",
            "model",
            "module",
            "encoder",
            "decoder",
            "training",
            "loss",
            "algorithm",
            "framework",
        )

        if self._looks_like_metadata_section(title, content):
            return "academic_guide"
        if any(keyword in title for keyword in academic_keywords):
            return "academic_guide"
        if any(keyword in title for keyword in code_keywords):
            return "code_alignment"
        if any(keyword in combined for keyword in ("equation", "formula", "attention", "fusion")):
            return "code_alignment"
        return "academic_guide"

    def _looks_like_metadata_section(self, title: str, content: str) -> bool:
        if "arxiv:" in title or "arxiv:" in content:
            return True
        metadata_terms = ("university", "institute", "author", "correspondence", "@")
        return any(term in content for term in metadata_terms)

    def _build_role_prompt(self, paper_section: PaperSection, section_mode: str) -> str:
        title = paper_section.title.lower()
        if "introduction" in title:
            role = "你现在的角色是科研领路人，请解释这段话的研究背景、问题定义和核心创新点。"
        elif "method" in title or "implementation" in title or "architecture" in title:
            role = (
                "你现在的角色是代码审计专家，请寻找这段公式或机制的具体代码落地，"
                "宁可拒绝硬凑，也不要给出幻觉式映射。"
            )
        elif section_mode == "academic_guide":
            role = "你现在的角色是科研领路人，请提炼核心思想、理论动机和阅读重点。"
        else:
            role = "你现在的角色是代码审计专家，请核对实现链路、算子选择和张量形状。"
        return "\n".join([self._llm_client.get_react_agent_role_prompt(), role])

    def _should_refuse_alignment(
        self,
        paper_section: PaperSection,
        result: AlignmentResult,
        current_candidates: tuple[AlignmentCandidate, ...],
    ) -> bool:
        if result.match_type == "missing_implementation":
            return True
        if not current_candidates:
            return True
        if result.match_type == "strong_match":
            return False
        if self._has_direct_semantic_anchor(paper_section, current_candidates[0].code_evidence):
            return False
        return current_candidates[0].retrieval_score < 0.03 and result.alignment_score < 0.72

    def _has_direct_semantic_anchor(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> bool:
        paper_text = paper_section.combined_text.lower()
        code_text = evidence.combined_text.lower()
        anchors = (
            "attention",
            "softmax",
            "encoder",
            "decoder",
            "fusion",
            "graph",
            "loss",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "optimizer",
            "momentum",
        )
        return any(anchor in paper_text and anchor in code_text for anchor in anchors)

    def _build_graceful_refusal_result(
        self,
        paper_section: PaperSection,
        *,
        reason: str,
        fallback_candidate: AlignmentCandidate | None = None,
    ) -> AlignmentResult:
        evidence = fallback_candidate.code_evidence if fallback_candidate is not None else None
        code_file_name = evidence.file_name if evidence is not None else "未定位到直接实现"
        code_snippet = (
            evidence.code_snippet if evidence is not None else "# 当前未定位到直接对应的源码\n"
        )
        start_line = evidence.start_line if evidence is not None else 1
        end_line = evidence.end_line if evidence is not None else 1
        return AlignmentResult(
            paper_section_title=paper_section.title,
            code_file_name=code_file_name,
            alignment_score=0.28,
            match_type="missing_implementation",
            analysis=self._build_translation_fallback(paper_section),
            improvement_suggestion="建议优先回看上游逻辑设计、配置入口和第三方依赖，再决定是否继续追源码。",
            retrieval_score=(
                fallback_candidate.retrieval_score if fallback_candidate is not None else 0.0
            ),
            semantic_evidence=self._build_core_points_fallback(paper_section),
            research_supplement=self._build_term_glossary(paper_section),
            evidence_level="弱关联：现有候选不足以支撑直接代码落地结论。",
            operator_alignment="当前未见能直接承接该段公式或机制的算子实现。",
            shape_alignment="当前未见足够可靠的张量形状处理证据。",
            highlighted_line_numbers=(start_line,) if evidence is not None else (),
            code_snippet=code_snippet,
            code_language=evidence.language if evidence is not None else "python",
            code_start_line=start_line,
            code_end_line=end_line,
            retrieval_plan="章节预审通过，但代码证据不足，因此我切回论文讲解模式。",
            implementation_chain="",
            reflection="在证据不足时，我选择拒绝强配对，避免把不相关代码误报成论文实现。",
            confidence_note=reason,
            needs_manual_review=True,
        )

    def _build_local_fallback_result(
        self,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        step_traces: tuple[StepExecutionTrace, ...],
        current_plan: ExecutionPlan,
    ) -> AlignmentResult:
        candidate = current_candidates[0]
        evidence = candidate.code_evidence
        plan_steps = current_plan.steps or tuple(trace.step for trace in step_traces)
        symbol = evidence.symbol_name or evidence.block_type
        return AlignmentResult(
            paper_section_title=paper_section.title,
            code_file_name=evidence.file_name,
            alignment_score=0.48,
            match_type="partial_match",
            analysis=self._build_translation_fallback(paper_section),
            improvement_suggestion="建议继续沿当前逻辑块向上追踪定义与配置来源。",
            retrieval_score=candidate.retrieval_score,
            semantic_evidence=self._build_core_points_fallback(paper_section),
            research_supplement=self._build_term_glossary(paper_section),
            evidence_level=self._derive_evidence_level(evidence, False),
            operator_alignment=self._build_operator_alignment(paper_section, evidence),
            shape_alignment=self._build_shape_alignment(paper_section, evidence),
            highlighted_line_numbers=(
                (evidence.start_line,) if evidence.start_line <= evidence.end_line else ()
            ),
            code_snippet=evidence.code_snippet,
            code_language=evidence.language,
            code_start_line=evidence.start_line,
            code_end_line=evidence.end_line,
            retrieval_plan="\n".join(step.display_text for step in plan_steps),
            implementation_chain=(
                f"在当前仓库里，最接近这段机制的是 `{evidence.file_name}` "
                f"的 L{evidence.start_line}-L{evidence.end_line}。"
                f"这段代码围绕 `{symbol}` 展开，负责承接与论文片段最相近的模块职责。"
            ),
            reflection="这一轮没有拿到稳定的模型总结，我保留了最接近的代码证据。",
            confidence_note="模型本轮响应不稳定，我先给你一个本地兜底结论。",
            agent_observations=tuple(trace.observation for trace in step_traces),
            needs_manual_review=True,
            plan_steps=plan_steps,
            step_traces=step_traces,
        )

    def _build_academic_only_result(
        self,
        paper_section: PaperSection,
        *,
        role_prompt: str,
        mode_note: str,
        event_handler: AgentEventHandler | None = None,
    ) -> AlignmentResult:
        self._emit(
            event_handler,
            kind="action",
            message="translate_section",
            action_input={"mode": "academic_guide"},
        )
        self._emit(
            event_handler,
            kind="thought",
            message="我先把当前片段完整翻成中文，确保后面的要点提炼建立在稳定译文上。",
        )
        analysis = self._generate_learning_translation(paper_section)
        self._emit(
            event_handler,
            kind="observation",
            message="中文译文阶段完成。",
        )
        self._emit(
            event_handler,
            kind="action",
            message="summarize_key_points",
            action_input={"mode": "academic_guide"},
        )
        self._emit(
            event_handler,
            kind="thought",
            message="译文已经拿到，我现在只提炼学术重点，不讨论代码映射。",
        )
        semantic_evidence = self._generate_learning_core_points(
            paper_section,
            translation=analysis,
        )
        self._emit(
            event_handler,
            kind="observation",
            message="核心要点提炼完成。",
        )
        self._emit(
            event_handler,
            kind="action",
            message="build_glossary",
            action_input={"mode": "academic_guide"},
        )
        self._emit(
            event_handler,
            kind="thought",
            message="接下来补术语百科，把片段里最关键的专门概念讲清楚。",
        )
        research_supplement = self._generate_learning_glossary(
            paper_section,
            translation=analysis,
            core_points=semantic_evidence,
        )
        self._emit(
            event_handler,
            kind="observation",
            message="术语百科整理完成。",
        )
        improvement_suggestion = "建议先读懂该段的研究动机，再跳到方法或实现章节进行代码深度对齐。"

        return AlignmentResult(
            paper_section_title=paper_section.title,
            code_file_name="未定位到本地实现",
            alignment_score=0.35,
            match_type="missing_implementation",
            analysis=analysis,
            improvement_suggestion=improvement_suggestion,
            retrieval_score=0.0,
            semantic_evidence=semantic_evidence,
            research_supplement=research_supplement,
            evidence_level="弱关联：当前仅有论文语义，没有本地源码支撑。",
            operator_alignment="未定位到源码，因此暂时无法把论文公式映射到具体算子。",
            shape_alignment="未定位到源码，因此暂时无法核对张量形状与输入输出维度。",
            highlighted_line_numbers=(),
            code_snippet="# 当前未定位到对应源码\n",
            code_start_line=1,
            code_end_line=1,
            retrieval_plan=mode_note,
            implementation_chain="",
            reflection="在缺少代码证据时，我优先保证论文解释的完整性。",
            confidence_note="当前片段属于学术导读模式，我不会对它强行发起代码对齐。",
            needs_manual_review=True,
        )

    def _build_research_supplement(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence | None,
        has_code: bool,
    ) -> str:
        return self._build_term_glossary(paper_section)

    def _finalize_learning_result(
        self,
        result: AlignmentResult,
        paper_section: PaperSection,
    ) -> AlignmentResult:
        analysis = self._normalize_translation_text(result.analysis, paper_section)
        analysis = self._ensure_chinese_translation(paper_section, analysis)
        semantic_evidence = self._normalize_core_points_text(
            result.semantic_evidence,
            paper_section,
        )
        semantic_evidence = self._ensure_learning_core_points(
            paper_section,
            semantic_evidence,
            translation=analysis,
        )
        research_supplement = self._normalize_glossary_text(
            result.research_supplement,
            paper_section,
        )
        research_supplement = self._ensure_learning_glossary(
            paper_section,
            research_supplement,
            translation=analysis,
            core_points=semantic_evidence,
        )
        return replace(
            result,
            analysis=analysis,
            semantic_evidence=semantic_evidence,
            research_supplement=research_supplement,
        )

    def _generate_learning_translation(self, paper_section: PaperSection) -> str:
        return self._ensure_chinese_translation(paper_section, "")

    def _ensure_chinese_translation(self, paper_section: PaperSection, text: str) -> str:
        if (
            text
            and not self._is_english_heavy(text)
            and not self._looks_like_empty_learning_text(text)
        ):
            return text

        payload = self._llm_client.generate_json(
            system_prompt=build_translation_json_system_prompt(),
            user_prompt=f"【待翻译论文片段】{paper_section.content}",
            temperature=0.0,
            max_tokens=1200,
        )
        if isinstance(payload, dict):
            translated = str(payload.get("translation", "")).strip()
            if self._is_acceptable_translation(translated, paper_section.content):
                return translated

        generate_text = getattr(self._llm_client, "generate_text", None)
        if callable(generate_text):
            translated_text = generate_text(
                system_prompt=build_translation_text_system_prompt(),
                user_prompt=f"【待翻译论文片段】{paper_section.content}",
                temperature=0.0,
                max_tokens=1400,
            )
            translated_text = translated_text.strip()
            if self._is_acceptable_translation(translated_text, paper_section.content):
                return translated_text

            segmented_translation = self._translate_in_segments(
                paper_section.content,
                generate_text,
            )
            if segmented_translation:
                return segmented_translation

        return self._build_translation_fallback(paper_section)

    def _translate_in_segments(
        self,
        content: str,
        generate_text: Callable[..., str],
    ) -> str:
        segments = self._split_translation_segments(content)
        if len(segments) <= 1:
            return ""

        translated_segments: list[str] = []
        for segment in segments:
            translated = generate_text(
                system_prompt=build_translation_segment_system_prompt(),
                user_prompt=f"【待翻译片段】{segment}",
                temperature=0.0,
                max_tokens=500,
            ).strip()
            if not self._is_acceptable_translation(translated, segment):
                return ""
            translated_segments.append(translated)

        return "\n".join(translated_segments).strip()

    def _split_translation_segments(self, content: str) -> tuple[str, ...]:
        normalized = " ".join(content.replace("\n", " ").split()).strip()
        if not normalized:
            return ()

        parts = re.split(r"(?<=[.!?;:])\s+", normalized)
        simple_parts = tuple(part.strip() for part in parts if part.strip())
        if len(simple_parts) > 1:
            return simple_parts

        segments: list[str] = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= 220:
                current = candidate
                continue
            if current:
                segments.append(current)
            current = part
        if current:
            segments.append(current)
        return tuple(segments)

    def _is_acceptable_translation(self, translated_text: str, source_text: str) -> bool:
        cleaned = translated_text.strip()
        if self._looks_like_empty_learning_text(cleaned):
            return False

        chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in cleaned)
        ascii_letters = sum(char.isascii() and char.isalpha() for char in cleaned)
        normalized_source = " ".join(source_text.split()).strip().lower()
        normalized_translated = " ".join(cleaned.split()).strip().lower()

        if normalized_source and normalized_translated == normalized_source:
            return False
        if chinese_chars >= 20:
            return True
        if chinese_chars >= 8 and chinese_chars >= ascii_letters:
            return True
        return not self._is_english_heavy(cleaned)

    def _generate_learning_core_points(
        self,
        paper_section: PaperSection,
        *,
        translation: str,
    ) -> str:
        return self._ensure_learning_core_points(
            paper_section,
            "",
            translation=translation,
        )

    def _ensure_learning_core_points(
        self,
        paper_section: PaperSection,
        text: str,
        *,
        translation: str,
    ) -> str:
        normalized = self._normalize_core_points_text(text, paper_section)
        if text.strip() and normalized.count("- ") >= 3:
            return normalized

        payload = self._llm_client.generate_json(
            system_prompt=build_learning_core_points_system_prompt(),
            user_prompt=(f"【论文片段原文】{paper_section.content}\n\n【中文译文】{translation}"),
            temperature=0.1,
            max_tokens=900,
        )
        if isinstance(payload, dict):
            candidate = payload.get("semantic_evidence", "")
            if isinstance(candidate, list):
                items = [str(item).strip() for item in candidate if str(item).strip()]
                if len(items) >= 3:
                    return self._format_bullets(items[:3])
            repaired = self._normalize_core_points_text(str(candidate), paper_section)
            if repaired.count("- ") >= 3:
                return repaired

        return self._build_core_points_fallback(paper_section)

    def _generate_learning_glossary(
        self,
        paper_section: PaperSection,
        *,
        translation: str,
        core_points: str,
    ) -> str:
        return self._ensure_learning_glossary(
            paper_section,
            "",
            translation=translation,
            core_points=core_points,
        )

    def _ensure_learning_glossary(
        self,
        paper_section: PaperSection,
        text: str,
        *,
        translation: str,
        core_points: str,
    ) -> str:
        normalized = self._normalize_glossary_text(text, paper_section)
        if text.strip() and normalized.count("- ") >= 2:
            return normalized

        payload = self._llm_client.generate_json(
            system_prompt=build_learning_glossary_system_prompt(),
            user_prompt=(
                f"【论文片段原文】{paper_section.content}\n\n"
                f"【中文译文】{translation}\n\n"
                f"【核心要点】{core_points}"
            ),
            temperature=0.1,
            max_tokens=900,
        )
        if isinstance(payload, dict):
            candidate = payload.get("research_supplement", "")
            if isinstance(candidate, list):
                items = [str(item).strip() for item in candidate if str(item).strip()]
                if len(items) >= 2:
                    return self._format_bullets(items[:3])
            repaired = self._normalize_glossary_text(str(candidate), paper_section)
            if repaired.count("- ") >= 2:
                return repaired

        return self._build_term_glossary(paper_section)

    def _normalize_translation_text(self, raw_text: str, paper_section: PaperSection) -> str:
        parsed = self._try_parse_structured_text(raw_text)
        if isinstance(parsed, dict):
            for key in ("translation", "translated_text", "chinese_translation"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""
        cleaned = raw_text.strip()
        if not cleaned or cleaned.startswith("{") or cleaned.startswith("["):
            return ""
        if self._looks_like_empty_learning_text(cleaned):
            return ""
        return cleaned

    def _normalize_core_points_text(self, raw_text: str, paper_section: PaperSection) -> str:
        parsed = self._try_parse_structured_text(raw_text)
        if isinstance(parsed, dict):
            items = self._extract_core_points_from_mapping(parsed)
            if items:
                return self._format_bullets(items[:3])
            return self._build_core_points_fallback(paper_section)
        cleaned = raw_text.strip()
        if not cleaned or cleaned.startswith("{") or cleaned.startswith("["):
            return self._build_core_points_fallback(paper_section)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) >= 3 and all(line.startswith("-") for line in lines[:3]):
            return "\n".join(lines[:3])
        return self._build_core_points_fallback(paper_section)

    def _normalize_glossary_text(self, raw_text: str, paper_section: PaperSection) -> str:
        parsed = self._try_parse_structured_text(raw_text)
        if isinstance(parsed, dict):
            items = self._extract_glossary_items_from_mapping(parsed)
            if items:
                return self._format_bullets(items[:3])
            return self._build_term_glossary(paper_section)
        cleaned = raw_text.strip()
        if not cleaned or cleaned.startswith("{") or cleaned.startswith("["):
            return self._build_term_glossary(paper_section)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            return "\n".join(lines[:3])
        return self._build_term_glossary(paper_section)

    def _try_parse_structured_text(self, raw_text: str) -> object:
        cleaned = raw_text.strip()
        if not cleaned or cleaned[0] not in "{[":
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(cleaned)
            except (ValueError, SyntaxError, TypeError):
                continue
        return None

    def _extract_core_points_from_mapping(self, payload: dict) -> list[str]:
        core_items: list[str] = []
        for key in (
            "core_problem",
            "key_innovation",
            "theoretical_motivation",
            "technical_approach",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                core_items.append(value.strip())
            elif isinstance(value, list):
                core_items.extend(str(item).strip() for item in value if str(item).strip())
        return core_items

    def _extract_glossary_items_from_mapping(self, payload: dict) -> list[str]:
        glossary_items: list[str] = []
        related_concepts = payload.get("related_concepts")
        if isinstance(related_concepts, dict):
            for term, explanation in related_concepts.items():
                glossary_items.append(f"**{term}**：{explanation}")
        for key in ("historical_context", "significance"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                glossary_items.append(value.strip())
        return glossary_items

    def _format_bullets(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items if item)

    def _looks_like_empty_learning_text(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return True
        punctuation_only = cleaned.strip(":：-·•,，.。;；!?！？[]{}()（）")
        if not punctuation_only:
            return True
        chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in cleaned)
        ascii_letters = sum(char.isascii() and char.isalpha() for char in cleaned)
        if len(cleaned) <= 4:
            return True
        if chinese_chars == 0 and ascii_letters <= 4:
            return True
        return False

    def _is_english_heavy(self, text: str) -> bool:
        latin_chars = sum(char.isascii() and char.isalpha() for char in text)
        chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in text)
        return latin_chars > chinese_chars * 2

    def _build_translation_fallback(self, paper_section: PaperSection) -> str:
        summary = " ".join(paper_section.content.replace("\n", " ").split()).strip()
        if not summary:
            return "当前片段未提取到可供翻译的正文内容。"
        return (
            "当前模型这一轮没有稳定产出完整中文译文。"
            "建议重新点击该段重试；如果连续两次都出现这个情况，我会继续把翻译链单独拆出来。"
        )

    def _build_core_points_fallback(self, paper_section: PaperSection) -> str:
        title = paper_section.title.strip() or "当前片段"
        short_content = paper_section.content.replace("\n", " ").strip()
        if len(short_content) > 90:
            short_content = short_content[:90] + "..."
        return (
            f"- 这段内容首先在说明“{title}”对应的问题背景或任务目标。\n"
            f"- 作者真正想强调的是这段机制为什么重要：{short_content}\n"
            "- 阅读时最值得抓住的是：它解决了什么痛点、提出了什么关键思路，"
            "以及它和已有方法差在哪里。"
        )

    def _build_term_glossary(self, paper_section: PaperSection) -> str:
        glossary_map = {
            "grounding": (
                "Grounding 指把自然语言、视觉观察和环境状态真正对上号，"
                "也就是让模型知道一句话具体对应哪一部分场景或动作。"
            ),
            "topological map": (
                "Topological Map 是拓扑地图，更关注地点之间怎么连通，"
                "而不是精确几何坐标，适合做导航决策。"
            ),
            "on-the-fly": ("On-the-fly 可以理解成边走边算、现场生成，不是提前把所有结果都准备好。"),
            "dual-scale": (
                "Dual-scale 指双尺度，通常是在全局和局部两个粒度上同时建模，"
                "让系统既看整体路线，也看眼前细节。"
            ),
            "cross-modal": "Cross-modal 指跨模态，把语言、视觉、地图等不同类型的信息放在一起理解。",
            "encoder": "Encoder 是编码器，负责把原始输入整理成后续模块更容易处理的表征。",
            "decoder": "Decoder 是解码器，负责根据已有表征一步步生成动作、预测结果或最终输出。",
            "attention": "Attention 是注意力机制，本质上是在一堆信息里挑出当前最该关注的部分。",
            "topological": "Topological 强调连通关系和结构关系，而不是精确距离。",
        }
        text = paper_section.combined_text.lower()
        matched = [
            f"- **{term.title()}**：{explanation}"
            for term, explanation in glossary_map.items()
            if term in text
        ]
        unique_matched = list(dict.fromkeys(matched))[:3]
        if len(unique_matched) >= 2:
            return "\n".join(unique_matched)

        for term in self._infer_term_candidates(paper_section):
            if len(unique_matched) >= 3:
                break
            unique_matched.append(
                f"- **{term}**：这是本段阅读时值得额外注意的术语，"
                "建议结合上下文理解它在任务设定或方法机制中的具体含义。"
            )
        return "\n".join(unique_matched[:3])

    def _infer_term_candidates(self, paper_section: PaperSection) -> tuple[str, ...]:
        text = paper_section.combined_text.lower()
        candidates: list[str] = []
        for term in (
            "Embodied AI",
            "Unseen Environment",
            "Vision-Language Navigation",
            "Navigation",
            "Instruction",
            "Policy",
            "Transformer",
            "Map",
            "Agent",
        ):
            if term.lower() in text:
                candidates.append(term)
        if candidates:
            return tuple(dict.fromkeys(candidates))
        return ("Task Setting", "Core Mechanism")

    def _has_strong_source_grounding(
        self,
        result: AlignmentResult,
        evidence: CodeEvidence,
    ) -> bool:
        if result.match_type != "strong_match":
            return False
        if result.alignment_score < 0.78:
            return False
        if not result.implementation_chain.strip():
            return False
        return evidence.file_name != "未定位到本地实现"

    def _derive_evidence_level(self, evidence: CodeEvidence, has_definition: bool) -> str:
        if has_definition and evidence.symbol_name:
            return f"强关联：已定位到 {evidence.symbol_name} 的源码逻辑块。"
        if evidence.symbol_name:
            return f"中关联：命中了 {evidence.symbol_name} 的实现入口，但定义链仍需补追。"
        return "弱关联：当前只拿到局部代码片段，仍需人工补充上下文。"

    def _build_operator_alignment(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> str:
        paper_text = paper_section.combined_text.lower()
        code_text = evidence.code_snippet.lower()
        if "softmax" in paper_text and "softmax" in code_text:
            return "论文中的 softmax 机制在代码里有直接对应算子。"
        if any(term in code_text for term in ("q_proj", "k_proj", "v_proj", "attention", "attn")):
            return "代码里出现了注意力相关算子链，和论文机制有明显呼应。"
        if any(term in code_text for term in ("cat(", "concat", "fusion", "fuse")):
            return "代码里出现了特征融合算子，和论文描述的融合步骤存在对应关系。"
        return "当前只看到局部实现入口，算子级对应关系仍需要继续追踪。"

    def _build_shape_alignment(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> str:
        paper_text = paper_section.combined_text.lower()
        code_text = evidence.code_snippet.lower()
        shape_ops = ("reshape", "view(", "permute", "transpose", "flatten")
        if any(term in code_text for term in shape_ops):
            if "head" in paper_text or "dimension" in paper_text or "shape" in paper_text:
                return "代码里存在显式的张量重排 / 维度变换，和论文的形状描述相符。"
            return "代码里存在显式的张量重排，但论文片段本身没有给出足够细的维度说明。"
        if "shape" in paper_text or "dimension" in paper_text:
            return "论文片段提到了维度约束，但当前代码块里还没看到明确的形状处理。"
        return "当前论文片段没有强调维度细节，因此形状核对以局部上下文为主。"

    def _emit_plan(self, plan: ExecutionPlan, handler: AgentEventHandler | None) -> None:
        if handler is None:
            return
        handler(
            {
                "kind": "plan_update",
                "message": plan.rationale or "计划已更新。",
                "remaining_steps": tuple(step.display_text for step in plan.steps),
                "is_finished": plan.is_finished,
            }
        )

    def _emit(self, handler: AgentEventHandler | None, **payload: object) -> None:
        if handler is not None:
            handler(payload)
