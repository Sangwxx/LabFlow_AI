"""基于按需触发的 Plan-and-Execute / ReAct Agent。"""

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
    """我先拆计划，再把每一步交给执行器。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def create_plan(
        self,
        paper_section: PaperSection,
        *,
        project_structure: str,
    ) -> ExecutionPlan:
        """根据论文片段和文件树生成初始步骤。"""

        schema = {
            "rationale": "中文规划思路",
            "steps": [
                {
                    "description": "中文步骤描述",
                    "objective": "该步骤验证的目标",
                }
            ],
        }
        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你当前扮演 Planner。",
                    "请把任务拆成 2 到 4 个步骤，步骤要可由工具执行。",
                    f"只输出 JSON，格式为: {json.dumps(schema, ensure_ascii=False)}",
                ]
            ),
            user_prompt=f"""
【论文片段标题】{paper_section.title}
【论文片段内容】
{paper_section.content}

【项目文件树】
{project_structure}
""".strip(),
            temperature=0.1,
            max_tokens=900,
        )
        if not isinstance(payload, dict):
            return self._fallback_plan()

        steps = self._normalize_steps(payload.get("steps"))
        rationale = str(payload.get("rationale", "")).strip()
        if not steps:
            return self._fallback_plan(rationale=rationale)
        return ExecutionPlan(steps=steps, rationale=rationale)

    def _normalize_steps(self, raw_steps: object) -> tuple[PlanStep, ...]:
        """把模型返回的步骤收敛成稳定结构。"""

        if not isinstance(raw_steps, list):
            return ()

        normalized_steps: list[PlanStep] = []
        for index, raw_step in enumerate(raw_steps, start=1):
            if isinstance(raw_step, dict):
                description = str(raw_step.get("description", "")).strip()
                objective = str(raw_step.get("objective", "")).strip()
            else:
                description = str(raw_step).strip()
                objective = ""
            if not description:
                continue
            normalized_steps.append(
                PlanStep(
                    step_id=str(index),
                    description=description,
                    objective=objective,
                )
            )
        return tuple(normalized_steps)

    def _fallback_plan(self, *, rationale: str = "") -> ExecutionPlan:
        """给 Planner 一个稳定兜底。"""

        return ExecutionPlan(
            steps=(
                PlanStep("1", "扫描文件树，定位可能承载论文机制的模块", "缩小目标文件范围"),
                PlanStep("2", "读取关键代码段，核对层结构、控制流程或损失组合", "确认实现链"),
                PlanStep("3", "对照论文步骤与参数，输出最终实现链路分析", "形成结论"),
            ),
            rationale=rationale or "先定位模块，再读代码，最后核对论文逻辑。",
        )


class PlanAndExecuteExecutor:
    """我负责执行 Planner 给出的单个步骤。"""

    def __init__(
        self,
        llm_client: LLMClient,
        evidence_builder: EvidenceBuilder,
    ) -> None:
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
        """对单步任务跑 1 到 3 轮 Thought -> Action -> Observation。"""

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
        """返回项目树。"""

        return project_structure or "当前代码目录为空。"

    def read_code_segment(
        self,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        path: str,
        line_start: int,
        line_end: int,
    ) -> tuple[str, tuple[str, ...]]:
        """精确读取代码段。"""

        for evidence in code_evidences:
            if evidence.file_name != path:
                continue
            if evidence.start_line <= line_start and evidence.end_line >= line_end:
                relative_start = max(line_start - evidence.start_line, 0)
                relative_end = line_end - evidence.start_line + 1
                snippet = "\n".join(
                    evidence.code_snippet.splitlines()[relative_start:relative_end]
                ).strip()
                return (
                    snippet or evidence.code_snippet,
                    (self._candidate_id_from_evidence(evidence),),
                )

        for evidence in code_evidences:
            if evidence.file_name == path:
                return (
                    evidence.code_snippet,
                    (self._candidate_id_from_evidence(evidence),),
                )
        return ("没有找到指定代码段。", ())

    def llm_semantic_search(
        self,
        *,
        query: str,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        top_k: int = 4,
    ) -> tuple[str, tuple[AlignmentCandidate, ...]]:
        """按需做语义搜索，不提前全量索引。"""

        synthetic_section = PaperSection(
            title=paper_section.title,
            content=query,
            level=paper_section.level,
            page_number=paper_section.page_number,
            order=paper_section.order,
        )
        heuristic_candidates = self._evidence_builder.build_alignment_candidates_from_inputs(
            paper_sections=(synthetic_section,),
            code_evidences=code_evidences,
            top_k=1,
        )
        if not heuristic_candidates:
            return ("语义搜索没有找到相关代码。", ())

        shortlisted = self._shortlist_candidates(heuristic_candidates, top_k=top_k)
        rerank_payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你正在执行 llm_semantic_search 工具。",
                    "请从候选代码里选出最可能与查询语义相关的前若干项。",
                    '只输出 JSON，格式为 {"selected_indexes": [0, 1, ...]}。',
                ]
            ),
            user_prompt=self._build_search_rerank_prompt(query, shortlisted),
            temperature=0.1,
            max_tokens=500,
        )
        if not isinstance(rerank_payload, dict):
            rerank_payload = {"selected_indexes": list(range(min(top_k, len(shortlisted))))}

        selected_indexes = rerank_payload.get("selected_indexes", [])
        selected_candidates: list[AlignmentCandidate] = []
        if isinstance(selected_indexes, list):
            for raw_index in selected_indexes:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(shortlisted):
                    candidate = shortlisted[index]
                    if candidate not in selected_candidates:
                        selected_candidates.append(candidate)

        if not selected_candidates:
            selected_candidates = list(shortlisted[:top_k])

        observation_lines = []
        for candidate in selected_candidates:
            observation_lines.append(
                " | ".join(
                    [
                        candidate.code_evidence.file_name,
                        f"L{candidate.code_evidence.start_line}-L{candidate.code_evidence.end_line}",
                        f"召回分 {candidate.retrieval_score}",
                    ]
                )
            )
        return ("\n".join(observation_lines), tuple(selected_candidates))

    def _shortlist_candidates(
        self,
        heuristic_candidates: tuple[AlignmentCandidate, ...],
        *,
        top_k: int,
    ) -> tuple[AlignmentCandidate, ...]:
        """按代码段去重，保留最值得让模型再判断的候选。"""

        deduplicated: dict[str, AlignmentCandidate] = {}
        for candidate in heuristic_candidates:
            candidate_id = self._candidate_id(candidate)
            previous = deduplicated.get(candidate_id)
            if previous is None or candidate.retrieval_score > previous.retrieval_score:
                deduplicated[candidate_id] = candidate
        ordered = sorted(deduplicated.values(), key=lambda item: item.retrieval_score, reverse=True)
        return tuple(ordered[:top_k])

    def _build_search_rerank_prompt(
        self,
        query: str,
        candidates: tuple[AlignmentCandidate, ...],
    ) -> str:
        """给语义搜索工具做一次小范围 LLM 重排。"""

        candidate_blocks = []
        for index, candidate in enumerate(candidates):
            candidate_blocks.append(
                f"""
[候选 {index}]
文件: {candidate.code_evidence.file_name}
范围: L{candidate.code_evidence.start_line}-L{candidate.code_evidence.end_line}
代码:
{candidate.code_evidence.code_snippet}
                """.strip()
            )
        return f"""
【搜索查询】
{query}

【候选代码】
{chr(10).join(candidate_blocks)}
""".strip()

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
        """执行一次工具调用。"""

        if action == "list_project_structure":
            observation = self.list_project_structure(project_structure)
            return (
                {
                    "tool_name": "list_project_structure",
                    "tool_input": "查看项目结构",
                    "observation": observation,
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
            merged_candidates = self._merge_selected_candidates(current_candidates, candidate_ids)
            return (
                {
                    "tool_name": "read_code_segment",
                    "tool_input": f"{path}:{line_start}-{line_end}",
                    "observation": observation,
                },
                merged_candidates,
            )

        params = action_input if isinstance(action_input, dict) else {}
        query = str(params.get("query", "")).strip() or paper_section.combined_text
        observation, searched_candidates = self.llm_semantic_search(
            query=query,
            paper_section=paper_section,
            code_evidences=code_evidences,
            top_k=4,
        )
        merged_candidates = self._merge_candidates(current_candidates, searched_candidates)
        return (
            {
                "tool_name": "llm_semantic_search",
                "tool_input": query,
                "observation": observation,
            },
            merged_candidates,
        )

    def _merge_selected_candidates(
        self,
        current_candidates: tuple[AlignmentCandidate, ...],
        candidate_ids: tuple[str, ...],
    ) -> tuple[AlignmentCandidate, ...]:
        """从已选候选里筛出被 read_code 命中的项。"""

        if not candidate_ids:
            return current_candidates
        candidate_id_set = set(candidate_ids)
        selected = [
            candidate
            for candidate in current_candidates
            if self._candidate_id(candidate) in candidate_id_set
        ]
        return tuple(selected or current_candidates)

    def _merge_candidates(
        self,
        current_candidates: tuple[AlignmentCandidate, ...],
        new_candidates: tuple[AlignmentCandidate, ...],
    ) -> tuple[AlignmentCandidate, ...]:
        """合并新旧候选。"""

        merged: dict[str, AlignmentCandidate] = {}
        for candidate in (*current_candidates, *new_candidates):
            candidate_id = self._candidate_id(candidate)
            previous = merged.get(candidate_id)
            if previous is None or candidate.retrieval_score > previous.retrieval_score:
                merged[candidate_id] = candidate
        ordered = sorted(merged.values(), key=lambda item: item.retrieval_score, reverse=True)
        return tuple(ordered[:6])

    def _candidate_id(self, candidate: AlignmentCandidate) -> str:
        """稳定标识候选代码段。"""

        return self._candidate_id_from_evidence(candidate.code_evidence)

    def _candidate_id_from_evidence(self, evidence: CodeEvidence) -> str:
        """稳定标识代码证据。"""

        return f"{evidence.file_name}:{evidence.start_line}-{evidence.end_line}"

    def _build_executor_system_prompt(self) -> str:
        """告诉 Executor 如何做 Thought / Action / Observation。"""

        schema = {
            "thought": "当前步骤的思考",
            "action": "list_project_structure | read_code_segment | llm_semantic_search | finish",
            "action_input": "对象形式的工具入参",
            "final_observation": "当 action=finish 时，给出这一步的结论观察",
        }
        return "\n".join(
            [
                self._llm_client.get_react_agent_role_prompt(),
                "你当前扮演 Executor。",
                "你必须先给出 Thought，再选择一个 Action。",
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
        """给 Executor 当前步骤、候选和历史动作。"""

        candidate_lines = []
        for index, candidate in enumerate(current_candidates[:4]):
            candidate_lines.append(
                " | ".join(
                    [
                        f"候选 {index}",
                        candidate.code_evidence.file_name,
                        f"L{candidate.code_evidence.start_line}-L{candidate.code_evidence.end_line}",
                        f"召回分 {candidate.retrieval_score}",
                    ]
                )
            )

        history_lines = []
        for invocation in tool_invocations:
            history_lines.extend(
                [
                    f"工具: {invocation.tool_name}",
                    f"输入: {invocation.tool_input}",
                    f"观察: {invocation.observation}",
                ]
            )

        return f"""
【步骤】{step.display_text}
【论文片段标题】{paper_section.title}
【论文片段内容】
{paper_section.content}

【项目结构】
{project_structure}

【当前候选】
{chr(10).join(candidate_lines) or "暂无候选"}

【已执行历史】
{chr(10).join(history_lines) or "无"}
""".strip()

    def _emit(self, handler: AgentEventHandler | None, **payload: object) -> None:
        """把执行事件推给 UI。"""

        if handler is not None:
            handler(payload)


class PlanAndExecuteRePlanner:
    """我根据 Observation 决定继续还是收束。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def update_plan(
        self,
        plan: ExecutionPlan,
        trace: StepExecutionTrace,
    ) -> ExecutionPlan:
        """更新剩余计划。"""

        remaining_steps = plan.steps[1:]
        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你当前扮演 RePlanner。",
                    "请根据刚完成步骤的 Observation 判断要不要继续执行剩余计划。",
                    (
                        "只输出 JSON，格式为 "
                        '{"is_finished": true/false, "final_summary": "...", '
                        '"remaining_steps": [...]}。'
                    ),
                ]
            ),
            user_prompt=f"""
【原计划】
{chr(10).join(step.display_text for step in plan.steps)}

【刚完成步骤】
{trace.step.display_text}

【Thought】
{trace.thought}

【Action】
{trace.action}

【Observation】
{trace.observation}

【剩余步骤】
{chr(10).join(step.display_text for step in remaining_steps) or "无"}
""".strip(),
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

        updated_steps = self._normalize_steps(payload.get("remaining_steps"))
        if not updated_steps:
            updated_steps = remaining_steps
        return ExecutionPlan(
            steps=updated_steps,
            rationale=plan.rationale,
            is_finished=bool(payload.get("is_finished", False)) or not updated_steps,
            final_summary=str(payload.get("final_summary", "")).strip(),
        )

    def _normalize_steps(self, raw_steps: object) -> tuple[PlanStep, ...]:
        """把再规划后的步骤收敛成稳定结构。"""

        if not isinstance(raw_steps, list):
            return ()

        normalized_steps: list[PlanStep] = []
        for index, raw_step in enumerate(raw_steps, start=1):
            if isinstance(raw_step, dict):
                description = str(raw_step.get("description", "")).strip()
                objective = str(raw_step.get("objective", "")).strip()
            else:
                description = str(raw_step).strip()
                objective = ""
            if not description:
                continue
            normalized_steps.append(
                PlanStep(
                    step_id=str(index),
                    description=description,
                    objective=objective,
                )
            )
        return tuple(normalized_steps)


class PlanAndExecuteAgent:
    """真正按需触发的 Planner / Executor / RePlanner Agent。"""

    def __init__(self, llm_client=None, evidence_builder: EvidenceBuilder | None = None) -> None:
        self._llm_client = llm_client or LLMClient()
        self._evidence_builder = evidence_builder or EvidenceBuilder()
        self.planner = PlanAndExecutePlanner(self._llm_client)
        self.executor = PlanAndExecuteExecutor(self._llm_client, self._evidence_builder)
        self.replanner = PlanAndExecuteRePlanner(self._llm_client)

    def run(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        project_structure: str,
        event_handler: AgentEventHandler | None = None,
    ) -> AlignmentResult | None:
        """只有点击特定 Section 后才启动 Agent。"""

        if not code_evidences:
            return None

        current_candidates = self._evidence_builder.build_alignment_candidates_from_inputs(
            paper_sections=(paper_section,),
            code_evidences=code_evidences,
            top_k=1,
        )
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

        if not current_candidates:
            return None

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
        """把执行结果总结成最终结论。"""

        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你当前在输出 Final Answer。",
                    "请根据 Thought / Action / Observation 轨迹，给出最终实现链路分析。",
                    (
                        "只输出 JSON，格式字段包括：best_candidate_index、alignment_score、"
                        "match_type、analysis、implementation_chain、semantic_evidence、"
                        "highlighted_lines、improvement_suggestion。"
                    ),
                ]
            ),
            user_prompt=self._build_final_answer_prompt(
                paper_section=paper_section,
                current_candidates=current_candidates,
                step_traces=step_traces,
            ),
            temperature=0.1,
            max_tokens=1800,
        )
        if not isinstance(payload, dict):
            payload = {
                "best_candidate_index": 0,
                "alignment_score": 0.45,
                "match_type": "partial_match",
                "analysis": "当前模型没有稳定返回结构化结论，建议人工核对。",
                "implementation_chain": "当前实现链路尚未稳定收敛。",
                "semantic_evidence": "只拿到了局部代码证据，缺少完整结构化回答。",
                "highlighted_lines": [],
                "improvement_suggestion": "建议重新触发一次分析，或人工阅读候选代码段。",
            }

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
        )

    def _build_final_answer_prompt(
        self,
        *,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        step_traces: tuple[StepExecutionTrace, ...],
    ) -> str:
        """给 Final Answer 阶段执行轨迹和候选代码。"""

        trace_lines = []
        for trace in step_traces:
            trace_lines.extend(
                [
                    f"[Current Plan] {trace.step.display_text}",
                    f"[Thought] {trace.thought}",
                    f"[Action] {trace.action}",
                    f"[Observation] {trace.observation}",
                ]
            )

        candidate_blocks = []
        for index, candidate in enumerate(current_candidates):
            candidate_blocks.append(
                f"""
[候选 {index}]
文件: {candidate.code_evidence.file_name}
范围: L{candidate.code_evidence.start_line}-L{candidate.code_evidence.end_line}
召回分: {candidate.retrieval_score}
代码:
{candidate.code_evidence.code_snippet}
                """.strip()
            )

        return f"""
【论文片段标题】{paper_section.title}
【论文片段内容】
{paper_section.content}

【执行轨迹】
{chr(10).join(trace_lines) or "无"}

【候选代码】
{chr(10).join(candidate_blocks)}
""".strip()

    def _reflect(
        self,
        result: AlignmentResult,
        *,
        paper_section: PaperSection,
    ) -> AlignmentResult:
        """在输出前做自我审计。"""

        payload = self._llm_client.generate_json(
            system_prompt="\n".join(
                [
                    self._llm_client.get_react_agent_role_prompt(),
                    "你当前在做 Reflection。",
                    "请审计最终结论是否过度自信，若置信度低于 0.8 必须建议人工核对。",
                    (
                        '只输出 JSON，格式为 {"reflection": "...", '
                        '"final_confidence": 0.0, "confidence_note": "...", '
                        '"needs_manual_review": true/false}。'
                    ),
                ]
            ),
            user_prompt=f"""
【论文片段标题】{paper_section.title}
【论文片段内容】
{paper_section.content}

【当前结论】
{result.analysis}

【实现链路】
{result.implementation_chain}

【语义证据】
{result.semantic_evidence}
""".strip(),
            temperature=0.0,
            max_tokens=900,
        )
        if not isinstance(payload, dict):
            payload = {
                "reflection": "当前没有稳定拿到反思结果，建议人工核对。",
                "final_confidence": result.alignment_score,
                "confidence_note": "模型反思阶段未稳定返回，建议人工核对。",
                "needs_manual_review": True,
            }

        try:
            reflected_score = float(payload.get("final_confidence", result.alignment_score))
        except (TypeError, ValueError):
            reflected_score = result.alignment_score
        reflected_score = max(0.0, min(1.0, reflected_score))

        confidence_note = str(payload.get("confidence_note", "")).strip()
        needs_manual_review = bool(payload.get("needs_manual_review", False))
        if reflected_score < 0.8:
            needs_manual_review = True
            if not confidence_note:
                confidence_note = "我找到了相关代码，但在变量映射上存在歧义，建议人工核对。"

        return replace(
            result,
            alignment_score=reflected_score,
            reflection=str(payload.get("reflection", "当前缺少自我审计结论。")).strip(),
            confidence_note=confidence_note,
            needs_manual_review=needs_manual_review,
        )

    def _emit_plan(
        self,
        plan: ExecutionPlan,
        handler: AgentEventHandler | None,
    ) -> None:
        """把当前计划推给 UI。"""

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
        """统一派发事件。"""

        if handler is not None:
            handler(payload)
