"""首页快速导读相关的纯函数。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape

from labflow.clients.llm_client import LLMClient
from labflow.ui.paper_preview import LandingPaperPreview

ASCII_LETTER_PATTERN = re.compile(r"[A-Za-z]")
CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
UPPER_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9-]{1,9}\b")

TASK_RULES: tuple[tuple[str, str], ...] = (
    ("vision-and-language navigation", "视觉语言导航（VLN）"),
    ("visual language navigation", "视觉语言导航（VLN）"),
    ("language navigation", "语言导航"),
    ("object detection", "目标检测"),
    ("semantic segmentation", "语义分割"),
    ("image segmentation", "图像分割"),
    ("question answering", "问答"),
    ("retrieval", "检索"),
    ("recommendation", "推荐"),
    ("reasoning", "推理"),
    ("planning", "规划"),
)

CHALLENGE_RULES: tuple[tuple[str, str], ...] = (
    ("unseen environment", "未见环境下的泛化"),
    ("unseen environments", "未见环境下的泛化"),
    ("ground language", "语言与视觉场景之间的对齐"),
    ("language grounding", "语言 grounding 的细粒度定位"),
    ("explore the environment", "探索过程中的持续决策"),
    ("long-term action planning", "长程动作规划"),
    ("large action space", "大动作空间下的推理"),
    ("fine-grained", "细粒度理解"),
    ("cross-modal", "跨模态信息融合"),
)

METHOD_RULES: tuple[tuple[str, str], ...] = (
    ("dual-scale graph transformer", "双尺度图 Transformer"),
    ("topological map", "在线构建拓扑地图"),
    ("global action planning", "全局动作规划"),
    ("local observations", "局部观测编码"),
    ("global map", "全局地图编码"),
    ("graph transformer", "图 Transformer 编码"),
    ("cross-modal understanding", "跨模态理解"),
    ("cross-modal", "跨模态对齐"),
    ("memory", "显式记忆"),
    ("retrieval", "检索模块"),
    ("attention", "注意力机制"),
)

LOW_VALUE_PATTERNS: tuple[str, ...] = (
    "从现有信息看",
    "当前还没有",
    "中文导读",
    "信息流",
    "再决定要不要继续追到代码实现",
)
STOP_BENCHMARK_TOKENS = {"THIS", "THAT", "WITH", "FROM", "THE", "AND", "FOR", "ACT", "LOCAL"}


@dataclass(frozen=True)
class QuickGuideFacts:
    """快速导读兜底生成时抽取出的论文事实。"""

    task: str
    challenge_points: tuple[str, ...]
    method_points: tuple[str, ...]
    result_points: tuple[str, ...]
    benchmark_names: tuple[str, ...]


@dataclass(frozen=True)
class LandingQuickGuide:
    """导读页阅读报告数据。"""

    headline: str
    core_question: str
    core_conclusion: str
    contribution: str
    intro_path: str
    data_method: str
    analysis_flow: str
    takeaway: str
    limitation: str
    abstract_digest: str


@dataclass(frozen=True)
class LandingQuickGuideState:
    """首页快速导读运行时状态。"""

    guide: LandingQuickGuide | None = None
    hint: str | None = None


def build_landing_quick_guide(
    preview: LandingPaperPreview,
    *,
    llm_client: LLMClient | None = None,
) -> LandingQuickGuide:
    """优先使用模型生成中文导读，失败时退回本地兜底。"""

    if llm_client is not None:
        payload = llm_client.generate_json(
            system_prompt=_build_quick_guide_system_prompt(),
            user_prompt=_build_quick_guide_user_prompt(preview),
            temperature=0.1,
            max_tokens=2200,
        )
        guide = _parse_quick_guide_payload(payload)
        if guide is not None:
            return guide

    return _build_fallback_quick_guide(preview)


def coerce_landing_quick_guide(guide: object, preview: LandingPaperPreview) -> LandingQuickGuide:
    """把历史缓存或非中文结果兜底成当前版本可用的中文导读。"""

    candidate = LandingQuickGuide(
        headline=_clean_quick_guide_text(getattr(guide, "headline", "")),
        core_question=_clean_quick_guide_text(
            getattr(guide, "core_question", getattr(guide, "problem", ""))
        ),
        core_conclusion=_clean_quick_guide_text(
            getattr(guide, "core_conclusion", getattr(guide, "focus", ""))
        ),
        contribution=_clean_quick_guide_text(getattr(guide, "contribution", "")),
        intro_path=_clean_quick_guide_text(getattr(guide, "intro_path", "")),
        data_method=_clean_quick_guide_text(
            getattr(guide, "data_method", getattr(guide, "method", ""))
        ),
        analysis_flow=_clean_quick_guide_text(getattr(guide, "analysis_flow", "")),
        takeaway=_clean_quick_guide_text(getattr(guide, "takeaway", "")),
        limitation=_clean_quick_guide_text(getattr(guide, "limitation", "")),
        abstract_digest=_clean_quick_guide_text(
            getattr(guide, "abstract_digest", ""),
            max_length=520,
        ),
    )

    if _needs_chinese_fallback(candidate):
        return _build_fallback_quick_guide(preview)
    return candidate


def build_quick_guide_html(guide: LandingQuickGuide) -> str:
    """把导读报告渲染为单卡片 HTML。"""

    sections = (
        ("核心研究问题", guide.core_question),
        ("核心结论", guide.core_conclusion),
        ("本文贡献", guide.contribution),
        ("Introduction 怎么展开", guide.intro_path),
        ("方法与实验", guide.data_method),
        ("分析怎么组织", guide.analysis_flow),
        ("我能学到什么", guide.takeaway),
        ("还可以怎么往下挖", guide.limitation),
        ("摘要导读", guide.abstract_digest),
    )
    sections_html = "".join(_build_quick_guide_item(label, body) for label, body in sections)
    return (
        '<div class="guide-report-shell">'
        f'<div class="guide-report-headline">{escape(guide.headline)}</div>'
        '<div class="guide-report-stack">'
        f"{sections_html}"
        "</div>"
        "</div>"
    )


def _build_quick_guide_item(label: str, body: str) -> str:
    return (
        '<div class="guide-report-item">'
        f'<div class="guide-report-label">{escape(label)}</div>'
        f'<div class="guide-report-body">{escape(body)}</div>'
        "</div>"
    )


def _build_quick_guide_system_prompt() -> str:
    return """
你是一名科研论文导读助手。
请基于论文标题、作者和摘要，输出一个适合导读页展示的 JSON。

硬性要求：
1. 除论文标题、作者名、模型名、数据集名等专有名词外，其余解释性文字必须使用中文。
2. 不要照搬英文摘要原句，要写成中文导读。
3. 不要空话，不要营销口吻，不要夸大结论。
4. 语气像人类开发者或实验室同学在介绍这篇论文该怎么读。
5. 每个字段控制在 2-4 句，简洁、具体、可读，不要只写一句概括。
6. 只返回 JSON 对象，不要输出额外解释。
7. 所有字段都必须落到论文的任务、方法模块、实验对象或结果上，不能写泛化套话。
8. 不要出现“从现有信息看”“建议优先关注输入输出关系”“当前还没有拿到稳定结果”这类空洞表达。
9. 每个字段尽量遵守这个结构：
   - 第一句直接回答标题问题；
   - 第二句解释为什么、怎么做或靠什么支撑；
   - 如有必要，第三句补充阅读重点、实验验证或局限。
10. 不要写成提纲词组，不要只写一句话。

字段：
- headline: 一句话概括这篇论文值得怎么读
- core_question: 这篇论文的核心研究问题是什么
- core_conclusion: 这篇论文最后得出的核心结论是什么
- contribution: 这篇论文相对已有工作新增了哪些关键贡献
- intro_path: 这篇论文的 introduction 大致是怎么铺垫问题并引出方法的
- data_method: 论文用了什么方法、什么实验对象或评测设置，为什么这些东西重要
- analysis_flow: 实验和分析部分是怎么组织起来支撑结论的
- takeaway: 站在科研训练角度，这篇论文最值得学习的地方是什么
- limitation: 这篇论文目前还可能有哪些不足，后续还可以往哪里继续做
- abstract_digest: 用中文概括摘要的主要信息，适合放在“摘要导读”区域
""".strip()


def _build_quick_guide_user_prompt(preview: LandingPaperPreview) -> str:
    authors = " / ".join(preview.authors) if preview.authors else "未知作者"
    meta = "；".join(preview.meta_items) if preview.meta_items else "无额外元数据"
    return f"标题：{preview.title}\n作者：{authors}\n元数据：{meta}\n摘要：{preview.abstract}\n"


def _parse_quick_guide_payload(payload: dict | None) -> LandingQuickGuide | None:
    if not isinstance(payload, dict):
        return None

    headline = _clean_quick_guide_text(payload.get("headline"))
    core_question = _clean_quick_guide_text(payload.get("core_question"))
    core_conclusion = _clean_quick_guide_text(payload.get("core_conclusion"))
    contribution = _clean_quick_guide_text(payload.get("contribution"))
    intro_path = _clean_quick_guide_text(payload.get("intro_path"))
    data_method = _clean_quick_guide_text(payload.get("data_method"))
    analysis_flow = _clean_quick_guide_text(payload.get("analysis_flow"))
    takeaway = _clean_quick_guide_text(payload.get("takeaway"))
    limitation = _clean_quick_guide_text(payload.get("limitation"))
    abstract_digest = _clean_quick_guide_text(payload.get("abstract_digest"), max_length=520)
    if not all(
        (
            headline,
            core_question,
            core_conclusion,
            contribution,
            intro_path,
            data_method,
            analysis_flow,
            takeaway,
            limitation,
            abstract_digest,
        )
    ):
        return None

    return LandingQuickGuide(
        headline=headline,
        core_question=core_question,
        core_conclusion=core_conclusion,
        contribution=contribution,
        intro_path=intro_path,
        data_method=data_method,
        analysis_flow=analysis_flow,
        takeaway=takeaway,
        limitation=limitation,
        abstract_digest=abstract_digest,
    )


def _build_fallback_quick_guide(preview: LandingPaperPreview) -> LandingQuickGuide:
    facts = _extract_quick_guide_facts(preview)
    method_summary = "、".join(facts.method_points[:3]) if facts.method_points else "核心方法"
    challenge_summary = (
        "，重点难点在" + "、".join(facts.challenge_points[:2]) if facts.challenge_points else ""
    )
    benchmark_summary = (
        f"实验上可以重点看 {'、'.join(facts.benchmark_names[:3])} 这些基准。"
        if facts.benchmark_names
        else "实验上重点看指标是否真的覆盖了它宣称解决的问题。"
    )
    benchmark_names = (
        "、".join(facts.benchmark_names[:3]) if facts.benchmark_names else "主要评测基准"
    )
    challenge_summary = "、".join(facts.challenge_points[:2])
    method_summary = "、".join(facts.method_points[:3]) if facts.method_points else "核心方法"
    result_summary = "；".join(facts.result_points[:2])
    return LandingQuickGuide(
        headline=(
            f"先抓 {facts.task} 的问题边界，"
            f"再顺着 {method_summary} 去看它如何把整条研究链路搭起来。"
        ),
        core_question=(
            f"这篇论文聚焦 {facts.task}，核心想解决的是 {challenge_summary}。"
            "它面对的不是单一步骤分类问题，而是语言理解、视觉感知和行动决策必须连续配合的任务。"
            "如果不先把任务输入、环境约束和评测目标看清，后面的模型设计会很容易读散。"
        ),
        core_conclusion=(
            f"作者的核心结论是：通过 {method_summary}，"
            f"可以把理解、建图和决策更稳地串起来；{result_summary}。"
            "换句话说，这篇论文想证明的不是某个单点模块更强，而是整条研究链路被重新组织后会更有效。"
        ),
        contribution=(
            f"相对已有工作，这篇论文的主要贡献在于把 {facts.task} 拆成更清楚的子问题，"
            f"并用 {method_summary} 去分别覆盖这些环节。"
            "这种贡献方式的好处是，读者能够明确看到每个模块到底在补哪个旧方法短板，而不是只看到一个总名字。"
        ),
        intro_path=(
            "Introduction 可以按“任务背景 -> 现有方法不足 -> "
            "本文方法切入点 -> 实验验证”这条顺序去读。"
            f"读的时候重点看作者是怎么把 {challenge_summary} 引到自己方法上的。"
            "如果这一段铺垫读顺了，后面的模型结构图和实验表格会更容易对上。"
        ),
        data_method=(
            f"方法部分先抓 {method_summary} 这些模块分别负责什么；"
            f"实验部分再看 {benchmark_names} 这些基准为什么能验证它的主张。"
            "阅读时不要只记模块名称，还要顺手判断这些模块是不是确实对应了论文前面提出的任务难点。"
        ),
        analysis_flow=(
            "实验阅读建议按“主结果 -> 关键模块是否真的有效 -> "
            f"指标和成功率是否同步改善”这条线走；{benchmark_summary}"
            "这样读可以避免只盯着单个最优数字，而忽略整套论证是否真的闭环。"
        ),
        takeaway=(
            f"这篇论文最值得学的是：它没有把 {facts.task} 当成单一步骤，"
            "而是先拆问题，再让方法设计一一对应这些难点。"
            "如果你以后自己写论文，这种“问题拆解 -> 模块对应 -> 实验证明”的组织方式，"
            "比单纯记模型名更有参考价值。"
        ),
        limitation=(
            "从摘要层面看，当前还无法判断方法代价、泛化边界和失败案例覆盖得够不够。"
            "后续可以重点追踪消融实验、错误案例和更复杂场景下的稳定性。"
            "如果正文没有把这些地方补齐，那这篇论文的说服力其实还是有上限的。"
        ),
        abstract_digest=_build_fallback_abstract_digest(facts),
    )


def _clean_quick_guide_text(value: object, max_length: int = 420) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split()).strip("：:;；，,。 ")
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def _needs_chinese_fallback(guide: LandingQuickGuide) -> bool:
    texts = (
        guide.headline,
        guide.core_question,
        guide.core_conclusion,
        guide.contribution,
        guide.intro_path,
        guide.data_method,
        guide.analysis_flow,
        guide.takeaway,
        guide.limitation,
        guide.abstract_digest,
    )
    if not all(texts):
        return True
    return any(_is_probably_english_sentence(text) for text in texts) or any(
        _is_low_information_text(text) for text in texts
    )


def _is_probably_english_sentence(text: str) -> bool:
    ascii_count = len(ASCII_LETTER_PATTERN.findall(text))
    chinese_count = len(CHINESE_CHAR_PATTERN.findall(text))
    return ascii_count >= 12 and chinese_count == 0


def _is_low_information_text(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) < 12:
        return True
    return any(pattern in normalized for pattern in LOW_VALUE_PATTERNS)


def _extract_quick_guide_facts(preview: LandingPaperPreview) -> QuickGuideFacts:
    corpus = f"{preview.title}\n{preview.abstract}".lower()
    task = _pick_first_match(corpus, TASK_RULES) or "当前研究任务"
    challenge_points = _pick_matches(corpus, CHALLENGE_RULES, limit=3)
    method_points = _pick_matches(corpus, METHOD_RULES, limit=4)
    benchmark_names = _extract_benchmark_names(preview.abstract)
    result_points = _build_result_points(preview.abstract, benchmark_names)

    if not challenge_points:
        challenge_points = ("任务建模与泛化",)
    if not method_points:
        method_points = ("核心模块拆分", "关键表示建模")

    return QuickGuideFacts(
        task=task,
        challenge_points=challenge_points,
        method_points=method_points,
        result_points=result_points,
        benchmark_names=benchmark_names,
    )


def _pick_first_match(corpus: str, rules: tuple[tuple[str, str], ...]) -> str | None:
    for pattern, label in rules:
        if pattern in corpus:
            return label
    return None


def _pick_matches(
    corpus: str,
    rules: tuple[tuple[str, str], ...],
    *,
    limit: int,
) -> tuple[str, ...]:
    hits: list[str] = []
    for pattern, label in rules:
        if pattern in corpus and label not in hits:
            hits.append(label)
        if len(hits) >= limit:
            break
    return tuple(hits)


def _extract_benchmark_names(abstract: str) -> tuple[str, ...]:
    tokens = [
        token
        for token in UPPER_TOKEN_PATTERN.findall(abstract)
        if token not in STOP_BENCHMARK_TOKENS
    ]
    unique_tokens: list[str] = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
    return tuple(unique_tokens[:4])


def _build_result_points(abstract: str, benchmark_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized = abstract.lower()
    points: list[str] = []
    if "outperform" in normalized or "significantly" in normalized or "improves" in normalized:
        if benchmark_names:
            points.append(f"在 {'、'.join(benchmark_names[:3])} 等基准上取得提升")
        else:
            points.append("实验结果相对已有方法有明显提升")
    if "success rate" in normalized:
        points.append("重点指标里包含成功率")
    if not points:
        points.append("实验部分用于验证方法是否真的覆盖论文声称的能力")
    return tuple(points)


def _build_fallback_abstract_digest(facts: QuickGuideFacts) -> str:
    method_summary = "、".join(facts.method_points[:3])
    challenge_summary = "、".join(facts.challenge_points[:2])
    result_summary = "；".join(facts.result_points[:2])
    return (
        f"这篇论文面向 {facts.task}，核心难点在于 {challenge_summary}。"
        f"作者给出的主方法是 {method_summary}，试图把理解、建图和决策这几步串成一条完整链路。"
        f"{result_summary}。"
        "如果把摘要当成整篇论文的缩影来看，它真正想强调的是：这不是单一模块优化，而是整条方法链路的协同设计。"
    )
