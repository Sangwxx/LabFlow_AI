"""Engine / Prompts / Tools 重构后的运行时测试。"""

from labflow.reasoning.agent_engine import (
    PlanAndExecuteEngine,
    PlanAndExecuteExecutor,
    PlanAndExecutePlanner,
    PlanAndExecuteRePlanner,
)
from labflow.reasoning.agent_tools import ReasoningToolbox
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import (
    AlignmentCandidate,
    CodeEvidence,
    ExecutionPlan,
    PaperSection,
    PlanStep,
)


class FakeRuntimeLLMClient:
    """我把 Planner / Executor / RePlanner 的最小闭环锁死。"""

    def generate_json(self, *, system_prompt: str, user_prompt: str, **_: object) -> dict:
        if "Planner" in system_prompt:
            return {
                "rationale": "先看目录，再读代码。",
                "section_type": "method",
                "enabled_agents": ["translation", "reading_summary", "glossary", "code_grounding"],
                "code_focus": ["attention", "output projection"],
                "steps": [
                    {"description": "扫描目录锁定实现入口", "objective": "缩小范围"},
                    {"description": "读取候选代码段", "objective": "确认实现"},
                ],
            }
        if "Executor" in system_prompt:
            if "工具: list_project_structure" not in user_prompt:
                return {
                    "thought": "我先看项目结构。",
                    "action": "list_project_structure",
                    "action_input": {},
                }
            return {
                "thought": "目录已经足够清楚，这一步先结束。",
                "action": "finish",
                "action_input": {},
                "final_observation": "attention.py 是最像的入口。",
            }
        if "RePlanner" in system_prompt:
            if "扫描目录锁定实现入口" in user_prompt:
                return {
                    "is_finished": False,
                    "final_summary": "",
                    "remaining_steps": [{"description": "读取候选代码段", "objective": "确认实现"}],
                }
            return {
                "is_finished": True,
                "final_summary": "证据足够。",
                "remaining_steps": [],
            }
        raise AssertionError(f"未预期的提示词: {system_prompt}")

    def get_react_agent_role_prompt(self) -> str:
        return "你是一个代码审计专家。"


class FallbackOnlyRuntimeLLMClient:
    """当执行器拿不到结构化动作时，Engine 应尽快收束。"""

    def __init__(self) -> None:
        self.replanner_call_count = 0

    def generate_json(self, *, system_prompt: str, **_: object) -> dict | None:
        if "RePlanner" in system_prompt:
            self.replanner_call_count += 1
        return None

    def get_react_agent_role_prompt(self) -> str:
        return "你是一个代码审计专家。"


def test_tool_registry_exposes_expected_tools() -> None:
    """默认工具注册表应暴露当前 Agent 运行所需的全部工具。"""

    registry = ReasoningToolbox(EvidenceBuilder()).build_registry()

    assert registry.tool_names == (
        "list_project_structure",
        "read_code_segment",
        "llm_semantic_search",
        "find_definition",
    )


def test_engine_runs_with_registry_driven_executor() -> None:
    """Engine 应通过 while-loop 驱动步骤执行并保留轨迹。"""

    llm_client = FakeRuntimeLLMClient()
    evidence_builder = EvidenceBuilder()
    registry = ReasoningToolbox(evidence_builder).build_registry()
    engine = PlanAndExecuteEngine(
        planner=PlanAndExecutePlanner(llm_client),
        executor=PlanAndExecuteExecutor(
            llm_client,
            evidence_builder,
            tool_registry=registry,
        ),
        replanner=PlanAndExecuteRePlanner(llm_client),
    )
    section = PaperSection(
        title="3.1 Attention",
        content="The model uses q projection and output projection.",
        level=2,
        page_number=2,
        order=1,
    )
    candidates = (
        AlignmentCandidate(
            paper_section=section,
            code_evidence=CodeEvidence(
                file_name="models/attention.py",
                code_snippet="def build_attention(x):\n    q = q_proj(x)\n    return out_proj(q)\n",
                related_git_diff="",
                symbols=("build_attention", "q_proj", "out_proj"),
                commit_context=(),
                start_line=24,
                end_line=26,
            ),
            retrieval_score=0.91,
        ),
    )

    plan, traces, latest_candidates = engine.run(
        paper_section=section,
        project_structure="models / attention.py",
        code_evidences=tuple(candidate.code_evidence for candidate in candidates),
        current_candidates=candidates,
        role_prompt="你是一个代码审计专家。",
    )

    assert plan.is_finished is True
    assert plan.section_type == "method"
    assert plan.enabled_agents[-1] == "code_grounding"
    assert plan.code_focus == ("attention", "output projection")
    assert len(traces) == 2
    assert traces[0].tool_invocations[0].tool_name == "list_project_structure"
    assert latest_candidates[0].code_evidence.file_name == "models/attention.py"


def test_engine_stops_early_when_executor_falls_back_without_tool_calls() -> None:
    """执行器连续拿不到结构化动作时，不应继续空转 RePlanner。"""

    llm_client = FallbackOnlyRuntimeLLMClient()
    evidence_builder = EvidenceBuilder()
    registry = ReasoningToolbox(evidence_builder).build_registry()
    engine = PlanAndExecuteEngine(
        planner=PlanAndExecutePlanner(llm_client),
        executor=PlanAndExecuteExecutor(
            llm_client,
            evidence_builder,
            tool_registry=registry,
        ),
        replanner=PlanAndExecuteRePlanner(llm_client),
    )
    section = PaperSection(
        title="3.2 Global Action Planning",
        content="The coarse-scale encoder predicts actions over visited nodes.",
        level=2,
        page_number=4,
        order=1,
    )
    candidates = (
        AlignmentCandidate(
            paper_section=section,
            code_evidence=CodeEvidence(
                file_name="models/graph_utils.py",
                code_snippet="def calculate_vp_rel_pos_fts(x):\n    return x\n",
                related_git_diff="",
                symbols=("calculate_vp_rel_pos_fts",),
                commit_context=(),
                start_line=15,
                end_line=16,
            ),
            retrieval_score=0.42,
        ),
    )
    current_plan = ExecutionPlan(
        steps=(
            PlanStep("1", "扫描目录锁定实现入口", "缩小范围"),
            PlanStep("2", "读取候选代码段", "确认实现"),
        ),
        rationale="先看目录，再读代码。",
        section_type="method",
        enabled_agents=("translation", "reading_summary", "glossary", "code_grounding"),
        code_focus=("graph",),
    )

    plan, traces, latest_candidates = engine.run(
        paper_section=section,
        project_structure="models / graph_utils.py",
        code_evidences=tuple(candidate.code_evidence for candidate in candidates),
        current_candidates=candidates,
        role_prompt="你是一个代码审计专家。",
        current_plan=current_plan,
    )

    assert plan.is_finished is True
    assert plan.final_summary == "执行器当前没有稳定产出结构化工具动作，本轮先用已有候选直接生成结果。"
    assert len(traces) == 1
    assert traces[0].used_fallback is True
    assert traces[0].tool_invocations == ()
    assert latest_candidates == candidates
    assert llm_client.replanner_call_count == 0


def test_engine_finishes_with_timeout_plan_when_budget_is_exhausted() -> None:
    """达到执行预算后，Engine 应直接结束并交给上层兜底。"""

    llm_client = FakeRuntimeLLMClient()
    evidence_builder = EvidenceBuilder()
    registry = ReasoningToolbox(evidence_builder).build_registry()
    engine = PlanAndExecuteEngine(
        planner=PlanAndExecutePlanner(llm_client),
        executor=PlanAndExecuteExecutor(
            llm_client,
            evidence_builder,
            tool_registry=registry,
        ),
        replanner=PlanAndExecuteRePlanner(llm_client),
    )
    section = PaperSection(
        title="3.1 Attention",
        content="The model uses q projection and output projection.",
        level=2,
        page_number=2,
        order=1,
    )
    candidates = (
        AlignmentCandidate(
            paper_section=section,
            code_evidence=CodeEvidence(
                file_name="models/attention.py",
                code_snippet="def build_attention(x):\n    q = q_proj(x)\n    return out_proj(q)\n",
                related_git_diff="",
                symbols=("build_attention", "q_proj", "out_proj"),
                commit_context=(),
                start_line=24,
                end_line=26,
            ),
            retrieval_score=0.91,
        ),
    )

    plan, traces, latest_candidates = engine.run(
        paper_section=section,
        project_structure="models / attention.py",
        code_evidences=tuple(candidate.code_evidence for candidate in candidates),
        current_candidates=candidates,
        role_prompt="你是一个代码审计专家。",
        max_runtime_sec=0,
    )

    assert plan.is_finished is True
    assert plan.final_summary == "执行阶段达到时限，我先用已有候选收束结果。"
    assert traces == ()
    assert latest_candidates == candidates
