"""首页快速导读相关的纯函数。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape

from labflow.clients.llm_client import LLMClient
from labflow.ui.paper_preview import LandingPaperPreview

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？.!?])\s+")


@dataclass(frozen=True)
class LandingQuickGuide:
    """首页快速导读卡片的数据。"""

    headline: str
    problem: str
    method: str
    focus: str


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
    """优先使用模型生成导读，失败时退回本地摘要兜底。"""

    if llm_client is not None:
        payload = llm_client.generate_json(
            system_prompt=_build_quick_guide_system_prompt(),
            user_prompt=_build_quick_guide_user_prompt(preview),
            temperature=0.1,
            max_tokens=700,
        )
        guide = _parse_quick_guide_payload(payload)
        if guide is not None:
            return guide

    return _build_fallback_quick_guide(preview)


def build_quick_guide_html(guide: LandingQuickGuide) -> str:
    """把快速导读渲染为首页卡片 HTML。"""

    return (
        '<div class="quick-guide-shell">'
        '<div class="quick-guide-kicker">快速导读</div>'
        f'<div class="quick-guide-headline">{escape(guide.headline)}</div>'
        '<div class="quick-guide-grid">'
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
请基于论文标题、作者和摘要，用中文输出一个适合首页展示的快速导读 JSON。

要求：
1. 不要写空话，不要夸大。
2. 语气像人类开发者在给同学讲这篇论文先看什么。
3. 每个字段控制在 1-2 句，简洁、具体、可读。
4. 只返回 JSON 对象，不要输出额外解释。

字段：
- headline: 一句话概括这篇论文值得怎么读
- problem: 这篇论文主要在解决什么问题
- method: 它的方法核心是什么
- focus: 第一次阅读时最值得先关注什么
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
    if not all((headline, problem, method, focus)):
        return None

    return LandingQuickGuide(
        headline=headline,
        problem=problem,
        method=method,
        focus=focus,
    )


def _build_fallback_quick_guide(preview: LandingPaperPreview) -> LandingQuickGuide:
    sentences = _split_sentences(preview.abstract)
    problem = (
        sentences[0]
        if sentences
        else f"这篇论文围绕《{preview.title}》展开，重点是先弄清楚任务设定和它要解决的核心瓶颈。"
    )
    method = (
        sentences[1]
        if len(sentences) >= 2
        else "建议先看摘要和方法部分，重点确认模型输入、核心模块以及最终输出是怎么串起来的。"
    )
    focus = "第一次阅读时，先抓任务背景、方法主线和实验验证，再决定哪些部分需要继续追到代码里看。"
    headline = "先用一屏内容抓住问题、方法和阅读重点，再进入工作区细看代码。"
    return LandingQuickGuide(
        headline=_clean_quick_guide_text(headline),
        problem=_clean_quick_guide_text(problem),
        method=_clean_quick_guide_text(method),
        focus=_clean_quick_guide_text(focus),
    )


def _split_sentences(text: str) -> tuple[str, ...]:
    normalized = " ".join(text.split())
    if not normalized:
        return ()
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(normalized) if part.strip()]
    if not parts:
        return (normalized,)
    return tuple(parts[:3])


def _clean_quick_guide_text(value: object, max_length: int = 140) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split()).strip("：:;；，,。 ")
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"
