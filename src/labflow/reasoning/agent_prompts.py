"""推理层 Prompt 模板。"""

from __future__ import annotations

import json

from labflow.reasoning.models import (
    AlignmentCandidate,
    AlignmentResult,
    PaperSection,
    PlanStep,
    StepExecutionTrace,
    ToolInvocation,
)


def build_planner_system_prompt(role_prompt: str) -> str:
    schema = {
        "rationale": "中文规划思路",
        "section_type": "metadata | academic_guide | method | mixed",
        "enabled_agents": [
            "translation",
            "reading_summary",
            "glossary",
            "code_grounding",
        ],
        "code_focus": ["源码检索焦点"],
        "steps": [{"description": "中文步骤", "objective": "目标"}],
    }
    return "\n".join(
        [
            role_prompt,
            "你当前扮演 Planner。",
            "你先判断当前论文片段需要启用哪些子 Agent，再决定是否生成代码对齐步骤。",
            "可启用的子 Agent 只有四个：translation、reading_summary、glossary、code_grounding。",
            (
                "如果片段主要是标题、作者信息、摘要背景、引言背景、结论总结，"
                "不要硬启用 code_grounding。"
            ),
            (
                "如果片段涉及 method、architecture、implementation、module、equation、"
                "loss、attention、fusion、map、policy 等机制内容，应优先启用 code_grounding。"
            ),
            "如果原文已经是中文，translation 可以关闭。",
            (
                "当启用 code_grounding 时，请把任务拆成 2 到 4 个可执行步骤，"
                "并给出 1 到 4 个源码检索焦点。"
            ),
            "不要把重点放在变量名映射上。",
            f"只输出 JSON，格式为: {json.dumps(schema, ensure_ascii=False)}",
        ]
    )


def build_planner_user_prompt(
    paper_section: PaperSection,
    *,
    project_structure: str,
    default_section_type: str,
    default_enabled_agents: tuple[str, ...],
    default_code_focus: tuple[str, ...],
) -> str:
    return (
        f"【论文片段】\n{paper_section.combined_text}\n\n"
        f"【项目文件树】\n{project_structure or '当前未提供文件树。'}\n\n"
        f"【预审建议】\n"
        f"- 默认片段类型: {default_section_type}\n"
        f"- 默认启用模块: {', '.join(default_enabled_agents)}\n"
        f"- 默认源码焦点: {', '.join(default_code_focus) or '无'}"
    )


def build_executor_system_prompt(role_prompt: str) -> str:
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


def build_executor_user_prompt(
    *,
    step: PlanStep,
    paper_section: PaperSection,
    project_structure: str,
    current_candidates: tuple[AlignmentCandidate, ...],
    tool_invocations: tuple[ToolInvocation, ...],
) -> str:
    candidates = (
        "\n".join(
            format_candidate_summary(candidate, index)
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


def build_replanner_system_prompt(role_prompt: str) -> str:
    return "\n".join(
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
    )


def build_replanner_user_prompt(
    plan_steps: tuple[PlanStep, ...],
    *,
    finished_step: PlanStep,
    thought: str,
    action: str,
    observation: str,
    remaining_steps: tuple[PlanStep, ...],
) -> str:
    return (
        f"【原计划】\n{chr(10).join(step.display_text for step in plan_steps)}\n\n"
        f"【刚完成步骤】{finished_step.display_text}\n"
        f"【Thought】{thought}\n"
        f"【Action】{action}\n"
        f"【Observation】{observation}\n\n"
        f"【剩余步骤】\n"
        f"{chr(10).join(step.display_text for step in remaining_steps) or '无'}"
    )


def format_candidate_summary(
    candidate: AlignmentCandidate,
    index: int | None = None,
) -> str:
    evidence = candidate.code_evidence
    prefix = f"候选{index} | " if index is not None else ""
    symbol = evidence.symbol_name or "未命名逻辑块"
    return (
        f"{prefix}{evidence.file_name} | "
        f"L{evidence.start_line}-L{evidence.end_line} | "
        f"{evidence.block_type} | {symbol} | "
        f"召回分 {candidate.retrieval_score:.3f}"
    )


def build_final_answer_system_prompt(role_prompt: str) -> str:
    return "\n".join(
        [
            role_prompt,
            "你当前在输出 Final Answer。",
            "你的输出定位是科研学习助手，不是错误日志或评分器。",
            "【中文译文】必须且只能写论文片段的中文翻译，不能讨论代码、模式切换或证据不足。",
            "【核心要点】必须是 3 条中文列表，提炼这段话最重要的 3 个学术观点。",
            "【术语百科】必须挑选 2 到 3 个专业英文词汇做通俗解释。",
            "只有当代码证据足够强时，才填写 implementation_chain 和 highlighted_lines。",
            "如果代码证据不够强，就把 implementation_chain 留空，不要硬凑源码解释。",
            "不要输出变量映射表、证据等级、自我审计这类内部术语。",
            (
                "只输出 JSON，字段包括 "
                '{"best_candidate_index", "alignment_score", "match_type", "analysis", '
                '"semantic_evidence", "research_supplement", "implementation_chain", '
                '"highlighted_lines", "improvement_suggestion"}。'
            ),
        ]
    )


def build_final_answer_user_prompt(
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
                f"逻辑块类型: {candidate.code_evidence.block_type}\n"
                f"符号: {candidate.code_evidence.symbol_name or '未命名逻辑块'}\n"
                f"Docstring: {candidate.code_evidence.docstring or '无'}\n"
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


def build_reflection_system_prompt(role_prompt: str) -> str:
    return "\n".join(
        [
            role_prompt,
            "你当前在做 Reflection。",
            "请审计最终结论是否过度自信。",
            "如果置信度低于 0.8，必须建议人工核对。",
            (
                "只输出 JSON，格式为 "
                '{"reflection": "...", "final_confidence": 0.0, '
                '"confidence_note": "...", "needs_manual_review": true/false}。'
            ),
        ]
    )


def build_reflection_user_prompt(
    paper_section: PaperSection,
    result: AlignmentResult,
) -> str:
    return (
        f"【论文片段】{paper_section.combined_text}\n\n"
        f"【当前结论】{result.analysis}\n\n"
        f"【实现链路】{result.implementation_chain}\n\n"
        f"【算子核对】{result.operator_alignment}\n\n"
        f"【形状核对】{result.shape_alignment}\n\n"
        f"【科研补完】{result.research_supplement}"
    )


def build_translation_json_system_prompt() -> str:
    return "\n".join(
        [
            "你是一个学术翻译助手。",
            "你的任务只有一个：把给定论文片段完整翻译成自然、准确的中文。",
            "不要总结，不要解释，不要输出术语百科，不要讨论代码。",
            '只输出 JSON，格式为 {"translation": "..."}。',
        ]
    )


def build_translation_text_system_prompt() -> str:
    return "\n".join(
        [
            "你是一个学术翻译助手。",
            "请把给定论文片段完整翻译成自然、准确、流畅的中文。",
            "不要总结，不要解释，不要补充背景，不要讨论代码。",
            "只输出中文译文正文。",
        ]
    )


def build_translation_segment_system_prompt() -> str:
    return "\n".join(
        [
            "你是一个学术翻译助手。",
            "请把给定英文片段准确翻译成自然、流畅的中文。",
            "不要总结，不要解释，只输出中文译文。",
        ]
    )


def build_learning_core_points_system_prompt() -> str:
    return "\n".join(
        [
            "你是一个耐心的科研学习助手。",
            "你的任务只有一个：提炼论文片段的 3 条核心要点。",
            "输出必须是中文列表，每条都要讲清这段话在解决什么问题或提出什么关键思想。",
            "不要讨论代码，不要解释为什么没有匹配到代码，不要输出 JSON 以外的多余文字。",
            '只输出 JSON，格式为 {"semantic_evidence": ["...", "...", "..."]}。',
        ]
    )


def build_learning_glossary_system_prompt() -> str:
    return "\n".join(
        [
            "你是一个耐心的科研学习助手。",
            "你的任务只有一个：解释论文片段中的 2 到 3 个关键专业术语。",
            "输出必须是中文列表，每条都要先写术语，再做通俗解释。",
            "不要讨论代码，不要输出实现链路，不要解释内部推理状态。",
            '只输出 JSON，格式为 {"research_supplement": ["...", "..."]}。',
        ]
    )
