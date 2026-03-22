"""推理层 Engine while-loop 运行时。"""

from __future__ import annotations

import time
from collections.abc import Callable

from labflow.clients.llm_client import LLMClient
from labflow.reasoning.agent_prompts import (
    build_executor_system_prompt,
    build_executor_user_prompt,
    build_planner_system_prompt,
    build_planner_user_prompt,
    build_replanner_system_prompt,
    build_replanner_user_prompt,
)
from labflow.reasoning.agent_tools import AgentToolContext, ReasoningToolbox, ToolRegistry
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import (
    AgentName,
    AlignmentCandidate,
    CodeEvidence,
    ExecutionPlan,
    PaperSection,
    PlanStep,
    StepExecutionTrace,
    ToolInvocation,
)

AgentEventHandler = Callable[[dict], None]


class PlanAndExecutePlanner:
    """先规划，再把执行交给 Engine。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def create_plan(
        self,
        paper_section: PaperSection,
        *,
        project_structure: str,
        role_prompt: str,
        default_section_type: str = "mixed",
        default_enabled_agents: tuple[AgentName, ...] = (),
        default_code_focus: tuple[str, ...] = (),
    ) -> ExecutionPlan:
        payload = self._llm_client.generate_json(
            system_prompt=build_planner_system_prompt(role_prompt),
            user_prompt=build_planner_user_prompt(
                paper_section,
                project_structure=project_structure,
                default_section_type=default_section_type,
                default_enabled_agents=default_enabled_agents,
                default_code_focus=default_code_focus,
            ),
            temperature=0.1,
            max_tokens=1000,
        )
        if not isinstance(payload, dict):
            return self._fallback_plan(
                default_section_type=default_section_type,
                default_enabled_agents=default_enabled_agents,
                default_code_focus=default_code_focus,
            )

        fallback = self._fallback_plan(
            default_section_type=default_section_type,
            default_enabled_agents=default_enabled_agents,
            default_code_focus=default_code_focus,
        )
        return ExecutionPlan(
            steps=self._normalize_steps(payload.get("steps")) or fallback.steps,
            rationale=str(payload.get("rationale", "")).strip() or fallback.rationale,
            section_type=str(payload.get("section_type", default_section_type)).strip()
            or default_section_type,
            enabled_agents=self._normalize_enabled_agents(
                payload.get("enabled_agents"),
                default_enabled_agents,
            ),
            code_focus=self._normalize_code_focus(
                payload.get("code_focus"),
                default_code_focus,
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

    def _normalize_enabled_agents(
        self,
        raw_agents: object,
        default_enabled_agents: tuple[AgentName, ...],
    ) -> tuple[AgentName, ...]:
        allowed = {"translation", "reading_summary", "glossary", "code_grounding"}
        if not isinstance(raw_agents, list):
            return default_enabled_agents
        normalized: list[AgentName] = []
        for item in raw_agents:
            agent_name = str(item).strip()
            if agent_name in allowed and agent_name not in normalized:
                normalized.append(agent_name)  # type: ignore[arg-type]
        return tuple(normalized) or default_enabled_agents

    def _normalize_code_focus(
        self,
        raw_code_focus: object,
        default_code_focus: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not isinstance(raw_code_focus, list):
            return default_code_focus
        normalized = [str(item).strip() for item in raw_code_focus if str(item).strip()]
        return tuple(normalized[:4]) or default_code_focus

    def _fallback_plan(
        self,
        *,
        default_section_type: str,
        default_enabled_agents: tuple[AgentName, ...],
        default_code_focus: tuple[str, ...],
    ) -> ExecutionPlan:
        return ExecutionPlan(
            steps=(
                PlanStep("1", "扫描文件树并锁定最可能的实现入口", "缩小目标范围"),
                PlanStep("2", "读取完整逻辑块并在必要时追踪定义源头", "形成实现链"),
                PlanStep("3", "核对算子、输入输出维度与论文步骤", "生成高可信结论"),
            ),
            rationale="先定入口，再追定义，最后核对算子和形状；没有源码就专注于论文讲解。",
            section_type=default_section_type,
            enabled_agents=default_enabled_agents,
            code_focus=default_code_focus,
        )


class PlanAndExecuteExecutor:
    """执行单个步骤，并通过 Tool Registry 调度动作。"""

    def __init__(
        self,
        llm_client: LLMClient,
        evidence_builder: EvidenceBuilder,
        *,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._evidence_builder = evidence_builder
        self._tool_registry = tool_registry or ReasoningToolbox(evidence_builder).build_registry()

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
        used_fallback = False

        for _ in range(max_actions):
            payload = self._llm_client.generate_json(
                system_prompt=build_executor_system_prompt(role_prompt),
                user_prompt=build_executor_user_prompt(
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
                used_fallback = True
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

            tool_result = self._tool_registry.execute(
                action,
                action_input,
                AgentToolContext(
                    paper_section=paper_section,
                    project_structure=project_structure,
                    code_evidences=code_evidences,
                    current_candidates=latest_candidates,
                ),
            )
            observation = tool_result.observation
            latest_candidates = tool_result.candidates
            tool_invocations.append(
                ToolInvocation(
                    tool_name=tool_result.tool_name,
                    tool_input=tool_result.tool_input,
                    observation=tool_result.observation,
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
                used_fallback=used_fallback,
            ),
            latest_candidates,
        )

    def _candidate_id(self, candidate: AlignmentCandidate) -> str:
        evidence = candidate.code_evidence
        return f"{evidence.file_name}:{evidence.start_line}-{evidence.end_line}"

    def _emit(self, handler: AgentEventHandler | None, **payload: object) -> None:
        if handler is not None:
            handler(payload)


class PlanAndExecuteRePlanner:
    """根据 Observation 更新计划。"""

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
            system_prompt=build_replanner_system_prompt(role_prompt),
            user_prompt=build_replanner_user_prompt(
                plan.steps,
                finished_step=trace.step,
                thought=trace.thought,
                action=trace.action,
                observation=trace.observation,
                remaining_steps=remaining_steps,
            ),
            temperature=0.1,
            max_tokens=700,
        )
        if not isinstance(payload, dict):
            return ExecutionPlan(
                steps=remaining_steps,
                rationale=plan.rationale,
                section_type=plan.section_type,
                enabled_agents=plan.enabled_agents,
                code_focus=plan.code_focus,
                is_finished=not remaining_steps,
                final_summary="",
            )

        return ExecutionPlan(
            steps=remaining_steps,
            rationale=plan.rationale,
            section_type=plan.section_type,
            enabled_agents=plan.enabled_agents,
            code_focus=plan.code_focus,
            is_finished=bool(payload.get("is_finished", False)) or not remaining_steps,
            final_summary=str(payload.get("final_summary", "")).strip(),
        )


class PlanAndExecuteEngine:
    """用最干净的 while-loop 驱动 Planner / Executor / RePlanner。"""

    def __init__(
        self,
        *,
        planner: PlanAndExecutePlanner,
        executor: PlanAndExecuteExecutor,
        replanner: PlanAndExecuteRePlanner,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._replanner = replanner

    def run(
        self,
        *,
        paper_section: PaperSection,
        project_structure: str,
        code_evidences: tuple[CodeEvidence, ...],
        current_candidates: tuple[AlignmentCandidate, ...],
        role_prompt: str,
        event_handler: AgentEventHandler | None = None,
        current_plan: ExecutionPlan | None = None,
        max_runtime_sec: float | None = None,
    ) -> tuple[ExecutionPlan, tuple[StepExecutionTrace, ...], tuple[AlignmentCandidate, ...]]:
        plan = current_plan or self._planner.create_plan(
            paper_section,
            project_structure=project_structure,
            role_prompt=role_prompt,
        )
        self._emit_plan(plan, event_handler)

        step_traces: list[StepExecutionTrace] = []
        active_plan = plan
        latest_candidates = current_candidates
        started_at = time.monotonic()

        while active_plan.steps and not active_plan.is_finished:
            if (
                max_runtime_sec is not None
                and time.monotonic() - started_at >= max_runtime_sec
            ):
                active_plan = self._build_timeout_plan(active_plan)
                self._emit_plan(active_plan, event_handler)
                break
            step = active_plan.steps[0]
            self._emit(
                event_handler,
                kind="current_plan",
                message=step.display_text,
                remaining_steps=tuple(item.display_text for item in active_plan.steps),
            )
            trace, latest_candidates = self._executor.execute(
                step,
                paper_section=paper_section,
                project_structure=project_structure,
                code_evidences=code_evidences,
                current_candidates=latest_candidates,
                role_prompt=role_prompt,
                event_handler=event_handler,
            )
            step_traces.append(trace)
            if trace.used_fallback and not trace.tool_invocations:
                active_plan = ExecutionPlan(
                    steps=(),
                    rationale=active_plan.rationale,
                    section_type=active_plan.section_type,
                    enabled_agents=active_plan.enabled_agents,
                    code_focus=active_plan.code_focus,
                    is_finished=True,
                    final_summary=(
                        "执行器当前没有稳定产出结构化工具动作，本轮先用已有候选直接生成结果。"
                    ),
                )
                self._emit_plan(active_plan, event_handler)
                break
            active_plan = self._replanner.update_plan(
                active_plan,
                trace,
                role_prompt=role_prompt,
            )
            if (
                max_runtime_sec is not None
                and time.monotonic() - started_at >= max_runtime_sec
            ):
                active_plan = self._build_timeout_plan(active_plan)
            self._emit_plan(active_plan, event_handler)

        return active_plan, tuple(step_traces), latest_candidates

    def _emit_plan(self, plan: ExecutionPlan, handler: AgentEventHandler | None) -> None:
        self._emit(
            handler,
            kind="plan_update",
            message=plan.rationale,
            remaining_steps=tuple(step.display_text for step in plan.steps),
        )

    def _emit(self, handler: AgentEventHandler | None, **payload: object) -> None:
        if handler is not None:
            handler(payload)

    def _build_timeout_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        return ExecutionPlan(
            steps=(),
            rationale=plan.rationale,
            section_type=plan.section_type,
            enabled_agents=plan.enabled_agents,
            code_focus=plan.code_focus,
            is_finished=True,
            final_summary="执行阶段达到时限，我先用已有候选收束结果。",
        )
