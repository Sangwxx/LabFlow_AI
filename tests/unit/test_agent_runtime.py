"""Engine / Prompts / Tools 重构后的运行时测试。"""

from labflow.reasoning.agent_engine import (
    PlanAndExecuteEngine,
    PlanAndExecuteExecutor,
    PlanAndExecutePlanner,
    PlanAndExecuteRePlanner,
)
from labflow.reasoning.agent_tools import ReasoningToolbox
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import AlignmentCandidate, CodeEvidence, PaperSection


class FakeRuntimeLLMClient:
    """我把 Planner / Executor / RePlanner 的最小闭环锁死。"""

    def generate_json(self, *, system_prompt: str, user_prompt: str, **_: object) -> dict:
        if "Planner" in system_prompt:
            return {
                "rationale": "先看目录，再读代码。",
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
    assert len(traces) == 2
    assert traces[0].tool_invocations[0].tool_name == "list_project_structure"
    assert latest_candidates[0].code_evidence.file_name == "models/attention.py"
