from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

from labflow.clients.llm_client import LLMClient
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
    """先把问题拆成 2 到 3 步，让执行器按需推进。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def create_plan(
        self,
        paper_section: PaperSection,
        *,
        project_structure: str,
    ) -> ExecutionPlan:
        schema = {
            "rationale": "中文规划思路",
            "steps": [{"description": "中文步骤", "objective": "目标"}],
        }
        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你当前扮演 Planner。",
                    "请把任务拆成 2 到 3 个可执行步骤。",
                    "优先围绕模块职责、控制流程和算法思想拆解。",
                    "不要把重点放在变量名映射上。",
                    f"只输出 JSON，格式为: {json.dumps(schema, ensure_ascii=False)}",
                ]
            ),
            user_prompt=(
                f"【论文片段】{paper_section.combined_text}\n\n"
                f"【项目文件树】\n"
                f"{project_structure or '当前未提供文件树。'}"
            ),
            temperature=0.1,
            max_tokens=900,
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
                PlanStep(
                    "1",
                    "扫描文件树，定位可能承载论文机制的模块",
                    "缩小目标文件范围",
                ),
                PlanStep(
                    "2",
                    "读取关键代码段，解释算法步骤如何落到实现中",
                    "确认实现链",
                ),
                PlanStep(
                    "3",
                    "若本地未见核心源码，则切换到学术解释模式",
                    "保证输出不断流",
                ),
            ),
            rationale="先定位模块，再解释实现；如果本地缺源码，就把论文讲清楚。",
        )


class PlanAndExecuteExecutor:
    """执行单步计划，按需调用工具并记录 Thought/Action/Observation。"""

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
                system_prompt=self._build_executor_system_prompt(),
                user_prompt=self._build_executor_user_prompt(
                    step=step,
                    paper_section=paper_section,
                    project_structure=project_structure,
                    current_candidates=latest_candidates,
                    tool_invocations=tuple(tool_invocations),
                ),
                temperature=0.1,
                max_tokens=900,
            )
            if not isinstance(payload, dict):
                payload = {
                    "thought": "当前模型没有稳定返回结构化动作，我先结束这一步。",
                    "action": "finish",
                    "action_input": {},
                    "final_observation": "当前步骤先用现有证据继续推进。",
                }

            thought = str(payload.get("thought", "继续核查实现链。")).strip()
            self._emit(
                event_handler,
                kind="thought",
                message=thought,
                step=step.display_text,
            )

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
        for evidence in code_evidences:
            if evidence.file_name != path:
                continue
            if evidence.start_line <= line_start and evidence.end_line >= line_end:
                start = max(line_start - evidence.start_line, 0)
                end = line_end - evidence.start_line + 1
                snippet = "\n".join(evidence.code_snippet.splitlines()[start:end]).strip()
                return snippet or evidence.code_snippet, (
                    self._candidate_id_from_evidence(evidence),
                )

        for evidence in code_evidences:
            if evidence.file_name == path:
                return evidence.code_snippet, (self._candidate_id_from_evidence(evidence),)

        return "没有找到指定代码段。", ()

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

    def _build_executor_system_prompt(self) -> str:
        schema = {
            "thought": "当前步骤的思考",
            "action": "list_project_structure | read_code_segment | llm_semantic_search | finish",
            "action_input": "对象形式的工具入参",
            "final_observation": "当 action=finish 时的阶段结论",
        }
        return "\n".join(
            [
                self._llm_client.get_react_agent_role_prompt(),
                "你当前扮演 Executor。",
                "你必须先给出 Thought，再选择一个 Action。",
                "Thought 要解释为什么这样查，不要只重复变量名。",
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
        prefix = f"候选 {index} | " if index is not None else ""
        evidence = candidate.code_evidence
        return (
            f"{prefix}{evidence.file_name} | "
            f"L{evidence.start_line}-L{evidence.end_line} | "
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
    """根据 Observation 决定是否继续执行剩余步骤。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def update_plan(
        self,
        plan: ExecutionPlan,
        trace: StepExecutionTrace,
    ) -> ExecutionPlan:
        remaining_steps = plan.steps[1:]
        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你当前扮演 RePlanner。",
                    "请根据刚完成步骤的 Observation 判断是否继续执行剩余计划。",
                    (
                        "只输出 JSON，格式为 "
                        '{"is_finished": true/false, '
                        '"final_summary": "...", '
                        '"remaining_steps": [...]}。'
                    ),
                ]
            ),
            user_prompt=(
                f"【原计划】\n"
                f"{chr(10).join(step.display_text for step in plan.steps)}\n\n"
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

    def __init__(
        self,
        llm_client=None,
        evidence_builder: EvidenceBuilder | None = None,
    ) -> None:
        self._llm_client = llm_client or LLMClient()
        self._evidence_builder = evidence_builder or EvidenceBuilder()
        self.planner = PlanAndExecutePlanner(self._llm_client)
        self.executor = PlanAndExecuteExecutor(
            self._llm_client,
            self._evidence_builder,
        )
        self.replanner = PlanAndExecuteRePlanner(self._llm_client)

    def run(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        project_structure: str,
        event_handler: AgentEventHandler | None = None,
    ) -> AlignmentResult | None:
        if not code_evidences:
            return self._build_academic_only_result(paper_section)

        current_candidates = self._evidence_builder.build_alignment_candidates_from_inputs(
            paper_sections=(paper_section,),
            code_evidences=code_evidences,
            top_k=4,
        )
        if not current_candidates:
            return self._build_academic_only_result(paper_section)

        plan = self.planner.create_plan(
            paper_section,
            project_structure=project_structure,
        )
        self._emit_plan(plan, event_handler)

        step_traces: list[StepExecutionTrace] = []
        current_plan = plan

        while current_plan.steps and not current_plan.is_finished:
            step = current_plan.steps[0]
            self._emit(
                event_handler,
                kind="current_plan",
                message=step.display_text,
                remaining_steps=tuple(item.display_text for item in current_plan.steps),
            )
            trace, current_candidates = self.executor.execute(
                step,
                paper_section=paper_section,
                project_structure=project_structure,
                code_evidences=code_evidences,
                current_candidates=current_candidates,
                event_handler=event_handler,
            )
            step_traces.append(trace)
            current_plan = self.replanner.update_plan(current_plan, trace)
            self._emit_plan(current_plan, event_handler)

        result = self._build_final_answer(
            paper_section=paper_section,
            current_candidates=current_candidates,
            step_traces=tuple(step_traces),
            current_plan=current_plan,
        )
        return self._reflect(result, paper_section=paper_section)

    def _build_final_answer(
        self,
        *,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        step_traces: tuple[StepExecutionTrace, ...],
        current_plan: ExecutionPlan,
    ) -> AlignmentResult:
        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你当前在输出 Final Answer。",
                    "不要只谈变量映射，要解释代码如何体现论文算法思想。",
                    "回答必须包含逻辑对齐和科研补完两部分。",
                    (
                        "只输出 JSON，字段包括 "
                        '{"best_candidate_index", "alignment_score", "match_type", '
                        '"analysis", "implementation_chain", "semantic_evidence", '
                        '"research_supplement", "highlighted_lines", '
                        '"improvement_suggestion"}。'
                    ),
                ]
            ),
            user_prompt=self._build_final_answer_prompt(
                paper_section,
                current_candidates,
                step_traces,
            ),
            temperature=0.1,
            max_tokens=1800,
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
        return replace(
            result,
            retrieval_plan="\n".join(step.display_text for step in plan_steps),
            plan_steps=plan_steps,
            step_traces=step_traces,
            agent_observations=tuple(trace.observation for trace in step_traces),
            research_supplement=(
                result.research_supplement
                or self._build_research_supplement(
                    paper_section,
                    current_candidates[selected_index].code_evidence,
                    True,
                )
            ),
        )

    def _build_final_answer_prompt(
        self,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        step_traces: tuple[StepExecutionTrace, ...],
    ) -> str:
        traces = (
            "\n".join(
                [
                    (
                        f"[Current Plan] {trace.step.display_text}\n"
                        f"[Thought] {trace.thought}\n"
                        f"[Action] {trace.action}\n"
                        f"[Observation] {trace.observation}"
                    )
                    for trace in step_traces
                ]
            )
            or "无"
        )
        candidates = "\n".join(
            [
                (
                    f"[候选 {index}]\n"
                    f"文件: {candidate.code_evidence.file_name}\n"
                    f"范围: L{candidate.code_evidence.start_line}"
                    f"-L{candidate.code_evidence.end_line}\n"
                    f"代码:\n{candidate.code_evidence.code_snippet}"
                )
                for index, candidate in enumerate(current_candidates)
            ]
        )
        return (
            f"【论文片段】{paper_section.combined_text}\n\n"
            f"【执行轨迹】\n{traces}\n\n"
            f"【候选代码】\n{candidates}"
        )

    def _reflect(
        self,
        result: AlignmentResult,
        *,
        paper_section: PaperSection,
    ) -> AlignmentResult:
        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你当前在做 Reflection。",
                    "请审计最终结论是否过度自信。",
                    "如果置信度低于 0.8，必须建议人工核对。",
                    (
                        "只输出 JSON，格式为 "
                        '{"reflection": "...", "final_confidence": 0.0, '
                        '"confidence_note": "...", "needs_manual_review": true/false}。'
                    ),
                ]
            ),
            user_prompt=(
                f"【论文片段】{paper_section.combined_text}\n\n"
                f"【当前结论】{result.analysis}\n\n"
                f"【实现链路】{result.implementation_chain}\n\n"
                f"【科研补完】{result.research_supplement}"
            ),
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
        symbol_preview = ", ".join(evidence.symbols[:5]) or "与该机制相关的核心调用"
        analysis = (
            f"{evidence.file_name} 的 "
            f"L{evidence.start_line}-L{evidence.end_line} "
            "是当前最接近论文机制的实现入口。"
        )
        return AlignmentResult(
            paper_section_title=paper_section.title,
            code_file_name=evidence.file_name,
            alignment_score=0.45,
            match_type="partial_match",
            analysis=analysis,
            improvement_suggestion="建议沿当前文件继续上追定义与配置来源，确认完整实现链。",
            retrieval_score=candidate.retrieval_score,
            semantic_evidence=(
                f"我优先命中了 {evidence.file_name}，因为它覆盖了符号 {symbol_preview}。"
            ),
            research_supplement=self._build_research_supplement(
                paper_section,
                evidence,
                True,
            ),
            highlighted_line_numbers=(
                (evidence.start_line,) if evidence.start_line <= evidence.end_line else ()
            ),
            code_snippet=evidence.code_snippet,
            code_language=evidence.language,
            code_start_line=evidence.start_line,
            code_end_line=evidence.end_line,
            retrieval_plan="\n".join(step.display_text for step in plan_steps),
            implementation_chain=(
                "我目前只能确认这段代码承担了与论文机制最相关的模块职责，"
                "完整算法链路仍建议结合上下文继续追踪。"
            ),
            reflection=(
                "这一轮没有拿到稳定的模型总结，我保留了最接近的代码证据，并建议你人工复核。"
            ),
            confidence_note="模型本轮响应不稳定，我先给你一个本地兜底结论。",
            agent_observations=tuple(trace.observation for trace in step_traces),
            needs_manual_review=True,
            plan_steps=plan_steps,
            step_traces=step_traces,
        )

    def _build_academic_only_result(self, paper_section: PaperSection) -> AlignmentResult:
        return AlignmentResult(
            paper_section_title=paper_section.title,
            code_file_name="未定位到本地实现",
            alignment_score=0.35,
            match_type="missing_implementation",
            analysis=(
                "当前没有在本地代码库中定位到能直接承接该论文片段的源码，因此我先切到学术解释模式。"
            ),
            improvement_suggestion=(
                "建议优先检查第三方依赖、配置驱动模块，或沿训练入口继续追踪调用链。"
            ),
            retrieval_score=0.0,
            semantic_evidence="本地未找到稳定候选代码，因此暂无可核验的实现证据。",
            research_supplement=self._build_research_supplement(
                paper_section,
                None,
                False,
            ),
            highlighted_line_numbers=(),
            code_snippet="# 当前未定位到对应源码\n",
            code_start_line=1,
            code_end_line=1,
            retrieval_plan="未发现可直接承接该论文片段的本地源码，已切换到学术解释模式。",
            implementation_chain="当前缺少本地代码证据，因此暂时无法构造可靠的实现链路。",
            reflection="在缺少代码证据时，我优先保证论文解释的完整性，而不是给出伪精确结论。",
            confidence_note="未命中本地源码，我已切换为论文阅读助手模式。",
            needs_manual_review=True,
        )

    def _build_research_supplement(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence | None,
        has_code: bool,
    ) -> str:
        summary = paper_section.content.replace("\n", " ").strip()
        if len(summary) > 140:
            summary = summary[:140] + "..."

        if has_code and evidence is not None:
            return (
                f"从论文语义看，这一段主要在说明“{paper_section.title}”对应的算法机制。"
                f"当前命中的 {evidence.file_name} 更像该机制的实现入口，"
                "建议把它和上游模块、配置项一起看。"
                "如果后续追踪仍缺失关键步骤，常见原因是："
                "核心逻辑被拆进基类、由配置开关动态注入，"
                "或者部分能力通过第三方库封装。"
            )

        return (
            f"学术解释模式：这段论文主要在讲“{paper_section.title}”。"
            f"就当前文字来看，它想强调的是：{summary}"
            "如果代码库里暂时找不到直接实现，常见原因包括："
            "该能力被第三方库封装、被基础类或公共模块间接实现，"
            "或者当前仓库并没有包含完整训练 / 推理链路。"
        )

    def _emit_plan(
        self,
        plan: ExecutionPlan,
        handler: AgentEventHandler | None,
    ) -> None:
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
