"""Plan-and-Execute / ReAct 对齐器测试。"""

from labflow.reasoning.agent_executor import PlanAndExecuteAgent, PlanAndExecuteExecutor
from labflow.reasoning.aligner import PaperCodeAligner, align_section
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import CodeEvidence, PaperSection


class FakePlanAndExecuteLLMClient:
    """我把 Planner、Executor、RePlanner、总结器和反思器整条链锁死。"""

    def get_react_agent_role_prompt(self) -> str:
        return "你是一个精通 PyTorch 和 Vision-Language Navigation 的源码审计专家。"

    def generate_json(self, *, system_prompt: str, user_prompt: str, **_: object) -> dict:
        if "Planner" in system_prompt:
            return {
                "rationale": "先看目录，再定位实现，最后核对论文机制。",
                "steps": [
                    {
                        "description": "扫描目录并定位可能承载注意力机制的文件",
                        "objective": "找到候选模块",
                    },
                    {
                        "description": "读取关键代码段并核对多头拆分与输出映射",
                        "objective": "确认实现链与参数",
                    },
                ],
            }

        if "Executor" in system_prompt:
            if "扫描目录并定位可能承载注意力机制的文件" in user_prompt:
                if "工具: list_project_structure" not in user_prompt:
                    return {
                        "thought": "我先看目录树，确认注意力实现最可能落在哪个文件。",
                        "action": "list_project_structure",
                        "action_input": {},
                    }
                return {
                    "thought": "目录里已经出现 attention.py，可以进入下一步。",
                    "action": "finish",
                    "action_input": {},
                    "final_observation": "attention.py 很可能承载论文里的多头注意力逻辑。",
                }

            if "读取关键代码段并核对多头拆分与输出映射" in user_prompt:
                if "工具: llm_semantic_search" not in user_prompt:
                    return {
                        "thought": (
                            "我先做一次按需语义搜索，看看哪段代码最像 q/k/v 投影和 8 头拆分。"
                        ),
                        "action": "llm_semantic_search",
                        "action_input": {
                            "query": (
                                "multi-head attention q k v projection "
                                "split 8 heads output projection"
                            )
                        },
                    }
                return {
                    "thought": "语义搜索已经锁定 attention.py，可以结束本步骤。",
                    "action": "finish",
                    "action_input": {},
                    "final_observation": "attention.py 中 q/k/v 投影、拆头和输出映射链路已经闭环。",
                }

        if "RePlanner" in system_prompt:
            if "扫描目录并定位可能承载注意力机制的文件" in user_prompt:
                return {
                    "is_finished": False,
                    "final_summary": "",
                    "remaining_steps": [
                        {
                            "description": "读取关键代码段并核对多头拆分与输出映射",
                            "objective": "确认实现链与参数",
                        }
                    ],
                }
            return {
                "is_finished": True,
                "final_summary": "证据已经足够形成实现链路分析。",
                "remaining_steps": [],
            }

        if "llm_semantic_search 工具" in system_prompt:
            return {"selected_indexes": [0]}

        if "Final Answer" in system_prompt:
            return {
                "best_candidate_index": 0,
                "alignment_score": 0.92,
                "match_type": "strong_match",
                "analysis": "该函数实现了论文里的多头注意力主路径。",
                "implementation_chain": (
                    "Algorithm 1 的 Step 1-3 对应 q/k/v 投影；"
                    "Step 4 对应 h=8 的拆头；"
                    "Step 5 对应 out_proj 输出映射。"
                ),
                "semantic_evidence": "代码先生成 q/k/v，再按 h=8 重排，最后经过 out_proj 输出。",
                "highlighted_lines": [24, 27, 29],
                "improvement_suggestion": "建议补充注释说明 head 数的来源。",
            }

        if "Reflection" in system_prompt or "反思器" in system_prompt:
            return {
                "reflection": "我复核后认为实现链路闭环，head=8 与论文一致，没有明显歧义。",
                "final_confidence": 0.88,
                "confidence_note": "",
                "needs_manual_review": False,
            }

        raise AssertionError(f"未预期的提示词: {system_prompt}")


class FakeLowConfidenceLLMClient(FakePlanAndExecuteLLMClient):
    """我专门模拟反思后降置信度的场景。"""

    def generate_json(self, *, system_prompt: str, user_prompt: str, **kwargs: object) -> dict:
        if "Reflection" in system_prompt or "反思器" in system_prompt:
            return {
                "reflection": "我找到了相关代码，但 alpha 和 beta 的变量映射仍有歧义。",
                "final_confidence": 0.62,
                "confidence_note": "我找到了相关代码，但在变量映射上存在歧义，建议人工核对。",
                "needs_manual_review": True,
            }
        return super().generate_json(system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)


class FakeUnavailableLLMClient:
    """我模拟整轮模型不可用，验证 Agent 也要优雅降级。"""

    def get_react_agent_role_prompt(self) -> str:
        return "你是一个源码审计专家。"

    def generate_json(self, **_: object) -> None:
        return None


def test_plan_and_execute_agent_builds_plan_and_traces() -> None:
    """Agent 应先规划，再按需执行，并保留执行轨迹。"""

    section = PaperSection(
        title="3.1 Multi-head Attention",
        content=(
            "The model projects q/k/v, splits them into 8 heads, "
            "merges them and applies output projection."
        ),
        level=2,
        page_number=3,
        order=1,
    )
    evidences = (
        CodeEvidence(
            file_name="models/attention.py",
            code_snippet=(
                "q = self.q_proj(x)\n"
                "k = self.k_proj(x)\n"
                "v = self.v_proj(x)\n"
                "heads = rearrange(q, 'b n (h d) -> b h n d', h=8)\n"
                "merged = heads.transpose(1, 2).reshape(batch, steps, -1)\n"
                "out = self.out_proj(merged)"
            ),
            related_git_diff="",
            symbols=("q_proj", "k_proj", "v_proj", "rearrange", "out_proj"),
            commit_context=("feat: add multi-head attention",),
            start_line=24,
            end_line=29,
        ),
    )

    result = PlanAndExecuteAgent(
        llm_client=FakePlanAndExecuteLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        evidences,
        project_structure="models / attention.py",
    )

    assert result is not None
    assert len(result.plan_steps) == 2
    assert len(result.step_traces) == 2
    assert result.step_traces[0].tool_invocations[0].tool_name == "list_project_structure"
    assert result.step_traces[1].tool_invocations[0].tool_name == "llm_semantic_search"
    assert "Step 4" in result.implementation_chain
    assert result.needs_manual_review is False


def test_align_section_marks_low_confidence_for_manual_review() -> None:
    """反思层把分数压到 8 分以下时，应诚实提示人工核对。"""

    section = PaperSection(
        title="3.2 Loss Weights",
        content="We set alpha = 0.70 and beta = 0.30 in the final loss.",
        level=2,
        page_number=4,
        order=9,
    )
    evidences = (
        CodeEvidence(
            file_name="trainer.py",
            code_snippet="alpha = 0.30\nbeta = 0.70\nloss = alpha * cls_loss + beta * reg_loss",
            related_git_diff="-alpha = 0.70\n+alpha = 0.30",
            symbols=("alpha", "beta", "loss"),
            commit_context=("fix: adjust loss weights",),
            start_line=12,
            end_line=14,
        ),
    )

    result = align_section(
        section,
        evidences,
        llm_client=FakeLowConfidenceLLMClient(),
        top_k=2,
        project_structure="trainer.py",
    )

    assert result is not None
    assert result.score_out_of_ten == 6.2
    assert result.needs_manual_review is True
    assert "建议人工核对" in result.confidence_note


def test_executor_tools_cover_project_structure_and_code_reading() -> None:
    """工具箱至少应能看目录、读代码段和做按需语义搜索。"""

    executor = PlanAndExecuteExecutor(
        llm_client=FakePlanAndExecuteLLMClient(),
        evidence_builder=EvidenceBuilder(),
    )
    evidences = (
        CodeEvidence(
            file_name="models/attention.py",
            code_snippet="def build_attention(x):\n    q = q_proj(x)\n    return out_proj(q)\n",
            related_git_diff="",
            symbols=("build_attention", "q_proj", "out_proj"),
            commit_context=(),
            start_line=24,
            end_line=26,
        ),
    )
    section = PaperSection(
        title="3.1 Attention",
        content="The model uses q projection and output projection.",
        level=2,
        page_number=2,
        order=1,
    )

    project_structure = executor.list_project_structure("models / attention.py")
    snippet, _ = executor.read_code_segment(
        evidences,
        path="models/attention.py",
        line_start=24,
        line_end=25,
    )
    search_result, candidates = executor.llm_semantic_search(
        query="attention q projection output projection",
        paper_section=section,
        code_evidences=evidences,
    )

    assert "models / attention.py" in project_structure
    assert "def build_attention" in snippet
    assert candidates
    assert "models/attention.py" in search_result


def test_agent_falls_back_when_llm_is_unavailable() -> None:
    """模型完全不可用时，Agent 也应该给出本地兜底结果，而不是崩溃。"""

    section = PaperSection(
        title="3.3 Attention",
        content="The model projects q and applies output projection.",
        level=2,
        page_number=3,
        order=2,
    )
    evidences = (
        CodeEvidence(
            file_name="models/attention.py",
            code_snippet="q = self.q_proj(x)\nout = self.out_proj(q)",
            related_git_diff="",
            symbols=("q_proj", "out_proj"),
            commit_context=(),
            start_line=24,
            end_line=25,
        ),
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeUnavailableLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        evidences,
        project_structure="models / attention.py",
    )

    assert result is not None
    assert result.match_type == "partial_match"
    assert result.needs_manual_review is True


def test_aligner_class_keeps_compatibility_entrypoint() -> None:
    """兼容入口仍可用，但底层已经换成按需 ReAct Agent。"""

    aligner = PaperCodeAligner(llm_client=FakePlanAndExecuteLLMClient())
    results = aligner.align_inputs(
        paper_sections=(
            PaperSection(
                title="4 Decoder",
                content="The decoder uses multi-head attention to mix graph context.",
                level=2,
                page_number=5,
                order=3,
            ),
        ),
        code_evidences=(
            CodeEvidence(
                file_name="attention.py",
                code_snippet="q = self.q_proj(x)\nout = self.out_proj(q)",
                related_git_diff="",
                symbols=("q_proj", "out_proj"),
                commit_context=("feat: add attention path",),
                start_line=24,
                end_line=25,
            ),
        ),
        project_structure="attention.py",
        top_k=2,
    )

    assert results
    assert results[0].analysis
