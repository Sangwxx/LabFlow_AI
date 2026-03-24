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
    """首页快速导读卡片的数据。"""

    headline: str
    problem: str
    method: str
    focus: str
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
            max_tokens=900,
        )
        guide = _parse_quick_guide_payload(payload)
        if guide is not None:
            return guide

    return _build_fallback_quick_guide(preview)


def coerce_landing_quick_guide(guide: object, preview: LandingPaperPreview) -> LandingQuickGuide:
    """把历史缓存或非中文结果兜底成当前版本可用的中文导读。"""

    if isinstance(guide, LandingQuickGuide):
        candidate = guide
    else:
        candidate = LandingQuickGuide(
            headline=_clean_quick_guide_text(getattr(guide, "headline", "")),
            problem=_clean_quick_guide_text(getattr(guide, "problem", "")),
            method=_clean_quick_guide_text(getattr(guide, "method", "")),
            focus=_clean_quick_guide_text(getattr(guide, "focus", "")),
            abstract_digest=_clean_quick_guide_text(
                getattr(guide, "abstract_digest", ""),
                max_length=220,
            ),
        )

    if _needs_chinese_fallback(candidate):
        return _build_fallback_quick_guide(preview)
    return candidate


def build_quick_guide_html(guide: LandingQuickGuide) -> str:
    """把快速导读渲染为卡片 HTML。"""

    return (
        '<div class="quick-guide-shell">'
        f'<div class="quick-guide-headline">{escape(guide.headline)}</div>'
        '<div class="quick-guide-stack">'
        f"{_build_quick_guide_item('这篇论文要解决什么', guide.problem)}"
        f"{_build_quick_guide_item('方法核心', guide.method)}"
        f"{_build_quick_guide_item('阅读时先关注什么', guide.focus)}"
        "</div>"
        "</div>"
    )


def _build_quick_guide_item(label: str, body: str) -> str:
    return (
        '<div class="quick-guide-item">'
        f'<div class="quick-guide-label">{escape(label)}</div>'
        f'<div class="quick-guide-body">{escape(body)}</div>'
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
5. 每个字段控制在 1-2 句，简洁、具体、可读。
6. 只返回 JSON 对象，不要输出额外解释。
7. problem、method、focus、abstract_digest 必须落到论文的任务、方法模块、
   实验对象或结果上，不能写泛化套话。
8. 不要出现“从现有信息看”“建议优先关注输入输出关系”“当前还没有拿到稳定结果”这类空洞表达。

字段：
- headline: 一句话概括这篇论文值得怎么读
- problem: 这篇论文主要在解决什么问题
- method: 它的方法核心是什么
- focus: 第一次阅读时最值得先关注什么
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
    problem = _clean_quick_guide_text(payload.get("problem"))
    method = _clean_quick_guide_text(payload.get("method"))
    focus = _clean_quick_guide_text(payload.get("focus"))
    abstract_digest = _clean_quick_guide_text(payload.get("abstract_digest"), max_length=220)
    if not all((headline, problem, method, focus, abstract_digest)):
        return None

    return LandingQuickGuide(
        headline=headline,
        problem=problem,
        method=method,
        focus=focus,
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
    return LandingQuickGuide(
        headline=f"先抓 {facts.task} 的任务目标，再看 {method_summary} 是怎么配合起来完成决策的。",
        problem=(
            f"这篇论文聚焦 {facts.task}{challenge_summary}。"
            "如果先不把任务输入、环境约束和评测目标看清，后面的方法部分会很容易读散。"
        ),
        method=(
            f"方法主线可以先概括成：{method_summary}。"
            "阅读时优先找清每个模块分别负责感知、建模还是决策，再看它们之间如何传递信息。"
        ),
        focus=(f"第一次阅读建议先看任务设定和方法总览，再顺着核心模块往下读；{benchmark_summary}"),
        abstract_digest=_build_fallback_abstract_digest(facts),
    )


def _clean_quick_guide_text(value: object, max_length: int = 140) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split()).strip("：:;；，,。 ")
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def _needs_chinese_fallback(guide: LandingQuickGuide) -> bool:
    texts = (
        guide.headline,
        guide.problem,
        guide.method,
        guide.focus,
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
    )
