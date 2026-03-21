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


class FakeStructuredStringLLMClient(FakeUnavailableLLMClient):
    """我专门模拟模型把结构化对象塞进字符串字段里的脏输出。"""

    def generate_json(self, *, system_prompt: str, **_: object) -> dict | None:
        if "你的任务只有一个：把给定论文片段完整翻译成自然、准确的中文。" in system_prompt:
            return {
                "translation": (
                    "{'paper_title': 'DUET', 'core_problem': 'Vision-Language Navigation "
                    "requires agents to follow natural language instructions in unseen "
                    "environments.'}"
                )
            }
        if "你的任务只有一个：提炼论文片段的 3 条核心要点。" in system_prompt:
            return {
                "semantic_evidence": (
                    "{'core_problem': 'Vision-Language Navigation requires agents to follow "
                    "natural language instructions in unseen environments.', "
                    "'key_innovation': 'The method uses a dual-scale architecture.', "
                    "'technical_approach': 'It combines local grounding with global planning.'}"
                )
            }
        if "你的任务只有一个：解释论文片段中的 2 到 3 个关键专业术语。" in system_prompt:
            return {
                "research_supplement": (
                    "{'related_concepts': {'Embodied AI': 'Embodied AI is about agents acting "
                    "in the world.', 'Unseen Environment': 'An unseen environment is a new "
                    "scene not observed during training.'}, 'significance': 'This setting tests "
                    "generalization.'}"
                )
            }
        return None


class FakeColonTranslationLLMClient(FakeUnavailableLLMClient):
    """我专门模拟模型把译文字段错误地只返回一个冒号。"""

    def generate_json(self, *, system_prompt: str, **_: object) -> dict | None:
        if "你的任务只有一个：提炼论文片段的 3 条核心要点。" in system_prompt:
            return {
                "semantic_evidence": (
                    "- 这段先定义了 Vision-Language Navigation 任务的基本挑战。\n"
                    "- 作者强调难点不只是理解语言，还包括在新环境中完成探索。\n"
                    "- 这为后续提出 DUET 的双尺度建模思路埋下了动机。"
                ),
            }
        if "你的任务只有一个：解释论文片段中的 2 到 3 个关键专业术语。" in system_prompt:
            return {
                "research_supplement": (
                    "- **Grounding**：把语言和场景、动作真正对应起来。\n"
                    "- **Topological Map**：更关注地点之间连通关系的地图表示。\n"
                    "- **Dual-Scale**：同时建模全局与局部两个尺度。"
                ),
            }
        if "你是一个学术翻译助手。" in system_prompt:
            return {
                "translation": (
                    "在未知环境中根据语言指令完成导航，对自主具身智能体来说是一个具有挑战性的问题。"
                )
            }
        return None


class FakeSequentialLearningLLMClient(FakeUnavailableLLMClient):
    """我记录调用顺序，确保学术导读模式真的按翻译 -> 要点 -> 术语执行。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_json(self, *, system_prompt: str, **_: object) -> dict | None:
        if "你是一个学术翻译助手。" in system_prompt:
            self.calls.append("translation")
            return {
                "translation": (
                    "在未知环境中根据语言指令完成导航，对自主具身智能体来说是一个具有挑战性的问题。"
                )
            }
        if "你的任务只有一个：提炼论文片段的 3 条核心要点。" in system_prompt:
            self.calls.append("core_points")
            return {
                "semantic_evidence": [
                    "这段先定义了 VLN 任务的基本挑战。",
                    "作者强调语言理解和环境探索必须同时成立。",
                    "这为后续提出双尺度方法提供了直接动机。",
                ]
            }
        if "你的任务只有一个：解释论文片段中的 2 到 3 个关键专业术语。" in system_prompt:
            self.calls.append("glossary")
            return {
                "research_supplement": [
                    "**Grounding**：把语言和场景、动作真正对应起来。",
                    "**Unseen Environment**：训练阶段没见过、测试时才遇到的新环境。",
                    "**Navigation**：根据指令持续决策并到达目标位置。",
                ]
            }
        return None


class FakeTextFallbackTranslationLLMClient(FakeUnavailableLLMClient):
    """我模拟 JSON 翻译失败，但纯文本翻译链可以成功救回来。"""

    def generate_json(self, *, system_prompt: str, **_: object) -> dict | None:
        if "你是一个学术翻译助手。" in system_prompt:
            return None
        if "你的任务只有一个：提炼论文片段的 3 条核心要点。" in system_prompt:
            return {
                "semantic_evidence": [
                    "这段先定义了任务背景与基本挑战。",
                    "作者强调语言理解与环境探索必须同时成立。",
                    "这为后续的方法设计提供了清晰动机。",
                ]
            }
        if "你的任务只有一个：解释论文片段中的 2 到 3 个关键专业术语。" in system_prompt:
            return {
                "research_supplement": [
                    "**Grounding**：把语言与场景、动作对应起来。",
                    "**Unseen Environment**：训练阶段没有见过的新环境。",
                ]
            }
        return None

    def generate_text(self, *, system_prompt: str, **_: object) -> str:
        if "你是一个学术翻译助手。" in system_prompt:
            return "在未知环境中根据语言指令完成导航，对自主具身智能体来说是一个具有挑战性的问题。"
        return ""


class FakeSegmentedTranslationLLMClient(FakeUnavailableLLMClient):
    """我模拟整段翻译失败，但分句翻译可以成功。"""

    def generate_json(self, *, system_prompt: str, **_: object) -> dict | None:
        if "你的任务只有一个：提炼论文片段的 3 条核心要点。" in system_prompt:
            return {
                "semantic_evidence": [
                    "这段定义了任务背景。",
                    "作者强调理解语言和探索环境要同时成立。",
                    "这为后续方法设计提供了动机。",
                ]
            }
        if "你的任务只有一个：解释论文片段中的 2 到 3 个关键专业术语。" in system_prompt:
            return {
                "research_supplement": [
                    "**Grounding**：让语言与环境实体建立对应关系。",
                    "**Navigation**：根据指令持续决策并到达目标。",
                ]
            }
        return None

    def generate_text(self, *, user_prompt: str, **_: object) -> str:
        if "【待翻译论文片段】" in user_prompt:
            return ""
        if "Following language instructions to navigate in unseen environments" in user_prompt:
            return "在未知环境中根据语言指令完成导航，是一个很有挑战性的问题。"
        if "The agent not only needs to ground languages in visual scenes" in user_prompt:
            return "智能体不仅需要把语言与视觉场景对齐，还需要探索环境以到达目标。"
        return ""


class FakeMixedChineseTranslationLLMClient(FakeUnavailableLLMClient):
    """我模拟模型返回夹带英文缩写的中文译文，系统也应当接受。"""

    def generate_json(self, *, system_prompt: str, **_: object) -> dict | None:
        if "你是一个学术翻译助手。" in system_prompt:
            return {
                "translation": (
                    "DUET 在 REVERIE、SOON 和 R2R 等 VLN 基准上显著优于现有方法，"
                    "同时提升了细粒度导航任务中的成功率。"
                )
            }
        if "你的任务只有一个：提炼论文片段的 3 条核心要点。" in system_prompt:
            return {
                "semantic_evidence": [
                    "作者强调方法在多个 VLN 基准上都优于现有方案。",
                    "这段突出了 DUET 的整体性能提升。",
                    "细粒度导航任务上的成功率提升也是重点结果。",
                ]
            }
        if "你的任务只有一个：解释论文片段中的 2 到 3 个关键专业术语。" in system_prompt:
            return {
                "research_supplement": [
                    "**VLN**：视觉语言导航任务。",
                    "**REVERIE**：目标导向导航基准。",
                ]
            }
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


def test_agent_prescreens_introduction_into_academic_guide_mode() -> None:
    """引言类片段应直接切到学术导读模式，而不是去代码库里乱撞。"""

    section = PaperSection(
        title="1 Introduction",
        content=(
            "Vision-and-language navigation studies how an embodied agent follows instructions."
        ),
        level=1,
        page_number=1,
        order=1,
    )
    evidences = (
        CodeEvidence(
            file_name="models/agent.py",
            code_snippet="class Agent:\n    pass\n",
            related_git_diff="",
            symbols=("Agent",),
            commit_context=(),
            start_line=1,
            end_line=2,
        ),
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeUnavailableLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        evidences,
        project_structure="models / agent.py",
    )

    assert result is not None
    assert result.match_type == "missing_implementation"
    assert "学术导读" in result.retrieval_plan
    assert "代码对齐" in result.confidence_note


def test_agent_refuses_to_force_irrelevant_method_alignment() -> None:
    """方法段若缺少可靠代码支撑，应诚实拒绝硬凑实现。"""

    section = PaperSection(
        title="3 Method",
        content="We derive a theoretical objective and discuss its optimization motivation.",
        level=2,
        page_number=4,
        order=5,
    )
    evidences = (
        CodeEvidence(
            file_name="utils/io.py",
            code_snippet="def load_yaml(path):\n    return path.read_text()\n",
            related_git_diff="",
            symbols=("load_yaml",),
            commit_context=(),
            start_line=3,
            end_line=4,
        ),
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeUnavailableLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        evidences,
        project_structure="utils / io.py",
    )

    assert result is not None
    assert result.match_type == "missing_implementation"
    assert "当前模型这一轮没有稳定产出完整中文译文" in result.analysis
    assert result.semantic_evidence.count("- ") == 3
    assert "无直接对应的算子实现" in result.confidence_note


def test_agent_sanitizes_structured_string_output_into_learning_sections() -> None:
    """即使模型把字典字符串塞进字段里，最终展示也必须是可读内容而不是原始对象。"""

    section = PaperSection(
        title="1 Introduction",
        content=(
            "Vision-Language Navigation requires agents to follow natural language "
            "instructions in unseen environments."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeStructuredStringLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
    )

    assert result is not None
    assert not result.analysis.startswith("{")
    assert not result.research_supplement.startswith("{")
    assert result.semantic_evidence.count("- ") == 3
    assert "Embodied AI" in result.research_supplement


def test_agent_repairs_colon_only_translation_output() -> None:
    """译文字段如果只剩一个冒号，我必须触发学习助手修复。"""

    section = PaperSection(
        title="Abstract",
        content=(
            "Following language instructions to navigate in unseen environments is a challenging "
            "problem for autonomous embodied agents."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeColonTranslationLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
    )

    assert result is not None
    assert result.analysis != ":"
    assert "具身智能体" in result.analysis
    assert result.semantic_evidence.count("- ") == 3
    assert result.research_supplement.count("- ") >= 2


def test_agent_uses_dedicated_translation_step_when_main_output_is_english() -> None:
    """主输出链即使给出英文，独立翻译链也要把译文救回来。"""

    section = PaperSection(
        title="Abstract",
        content=(
            "Following language instructions to navigate in unseen environments is a challenging "
            "problem for autonomous embodied agents."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeColonTranslationLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
    )

    assert result is not None
    assert "语言指令" in result.analysis
    assert "具身智能体" in result.analysis
    assert "Following language instructions" not in result.analysis


def test_agent_does_not_use_english_raw_text_as_translation_fallback() -> None:
    """当学习助手修复失败时，译文区也不该直接回退成英文原文。"""

    section = PaperSection(
        title="Abstract",
        content=(
            "Following language instructions to navigate in unseen environments is a challenging "
            "problem for autonomous embodied agents."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeUnavailableLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
    )

    assert result is not None
    assert "当前模型这一轮没有稳定产出完整中文译文" in result.analysis
    assert "Following language instructions" not in result.analysis


def test_agent_runs_learning_sections_in_sequence() -> None:
    """学术导读模式必须先翻译，再做要点，最后才做术语。"""

    llm_client = FakeSequentialLearningLLMClient()
    section = PaperSection(
        title="Abstract",
        content=(
            "Following language instructions to navigate in unseen environments is a challenging "
            "problem for autonomous embodied agents."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    result = PlanAndExecuteAgent(
        llm_client=llm_client,
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
    )

    assert result is not None
    assert llm_client.calls == ["translation", "core_points", "glossary"]
    assert "自主具身智能体" in result.analysis
    assert result.semantic_evidence.count("- ") == 3
    assert result.research_supplement.count("- ") >= 2


def test_agent_uses_plain_text_translation_fallback_when_json_translation_fails() -> None:
    """JSON 译文失败时，仍应切到纯文本翻译链拿回中文译文。"""

    section = PaperSection(
        title="1. Introduction",
        content=(
            "Following language instructions to navigate in unseen environments is a challenging "
            "problem for autonomous embodied agents."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeTextFallbackTranslationLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
    )

    assert result is not None
    assert "自主具身智能体" in result.analysis
    assert "当前模型这一轮没有稳定产出完整中文译文" not in result.analysis


def test_agent_uses_segment_translation_when_full_translation_fails() -> None:
    """整段翻译失败时，仍应退到分句翻译链。"""

    section = PaperSection(
        title="Abstract",
        content=(
            "Following language instructions to navigate in unseen environments is a challenging "
            "problem. The agent not only needs to ground languages in visual scenes but also "
            "explore the environment to reach its target."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeSegmentedTranslationLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
    )

    assert result is not None
    assert "在未知环境中根据语言指令完成导航" in result.analysis
    assert "探索环境以到达目标" in result.analysis
    assert "当前模型这一轮没有稳定产出完整中文译文" not in result.analysis


def test_academic_guide_mode_emits_multiple_trace_events() -> None:
    """学术导读模式也应显式展示翻译、要点、术语三个动作。"""

    events: list[dict] = []
    section = PaperSection(
        title="Abstract",
        content=(
            "Following language instructions to navigate in unseen environments is a challenging "
            "problem for autonomous embodied agents."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    PlanAndExecuteAgent(
        llm_client=FakeSequentialLearningLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
        event_handler=events.append,
    )

    action_messages = [
        str(event.get("message", "")) for event in events if event.get("kind") == "action"
    ]
    assert action_messages == [
        "translate_section",
        "summarize_key_points",
        "build_glossary",
    ]


def test_agent_accepts_chinese_translation_with_english_acronyms() -> None:
    """带英文缩写的中文译文不应再被误判成失败。"""

    section = PaperSection(
        title="Abstract",
        content=(
            "DUET significantly outperforms state-of-the-art methods on REVERIE, SOON and R2R."
        ),
        level=1,
        page_number=1,
        order=1,
    )

    result = PlanAndExecuteAgent(
        llm_client=FakeMixedChineseTranslationLLMClient(),
        evidence_builder=EvidenceBuilder(),
    ).run(
        section,
        (),
        project_structure="",
    )

    assert result is not None
    assert "显著优于现有方法" in result.analysis
    assert "当前模型这一轮没有稳定产出完整中文译文" not in result.analysis


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
