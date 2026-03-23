from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from labflow.clients.llm_client import LLMClient
from labflow.reasoning.agent_prompts import (
    build_learning_core_points_system_prompt,
    build_learning_glossary_system_prompt,
    build_translation_json_system_prompt,
    build_translation_segment_system_prompt,
    build_translation_text_system_prompt,
)
from labflow.reasoning.models import PaperSection

LearningEventHandler = Callable[[dict], None]


@dataclass(frozen=True)
class LearningOutputs:
    """学习助手三段式输出。"""

    translation: str
    core_points: str
    glossary: str


class TranslationAgent:
    """我只负责把论文片段稳定翻成中文。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def translate(self, paper_section: PaperSection) -> str:
        return self._ensure_chinese_translation(paper_section, "")

    def reuse_or_translate(self, paper_section: PaperSection, text: str) -> str:
        return self._ensure_chinese_translation(paper_section, text)

    def _ensure_chinese_translation(self, paper_section: PaperSection, text: str) -> str:
        if (
            text
            and not self._is_english_heavy(text)
            and not self._looks_like_empty_learning_text(text)
        ):
            return text

        try:
            payload = self._llm_client.generate_json(
                system_prompt=build_translation_json_system_prompt(),
                user_prompt=f"【待翻译论文片段】\n{paper_section.content}",
                temperature=0.0,
                max_tokens=1200,
            )
        except Exception:  # noqa: BLE001
            payload = None
        if isinstance(payload, dict):
            translated = str(payload.get("translation", "")).strip()
            if self._is_acceptable_translation(translated, paper_section.content):
                return translated

        generate_text = getattr(self._llm_client, "generate_text", None)
        if callable(generate_text):
            try:
                translated_text = generate_text(
                    system_prompt=build_translation_text_system_prompt(),
                    user_prompt=f"【待翻译论文片段】\n{paper_section.content}",
                    temperature=0.0,
                    max_tokens=1400,
                ).strip()
            except Exception:  # noqa: BLE001
                translated_text = ""
            if self._is_acceptable_translation(translated_text, paper_section.content):
                return translated_text

            segmented_translation = self._translate_in_segments(
                paper_section.content,
                generate_text,
            )
            if segmented_translation:
                return segmented_translation

        return self._build_translation_fallback(paper_section)

    def normalize_translation_text(self, raw_text: str) -> str:
        parsed = self._try_parse_structured_text(raw_text)
        if isinstance(parsed, dict):
            for key in ("translation", "translated_text", "chinese_translation"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""
        cleaned = raw_text.strip()
        if not cleaned or cleaned.startswith("{") or cleaned.startswith("["):
            return ""
        if self._looks_like_empty_learning_text(cleaned):
            return ""
        return cleaned

    def _translate_in_segments(
        self,
        content: str,
        generate_text: Callable[..., str],
    ) -> str:
        segments = self._split_translation_segments(content)
        if len(segments) <= 1:
            return ""

        translated_segments: list[str] = []
        for segment in segments:
            try:
                translated = generate_text(
                    system_prompt=build_translation_segment_system_prompt(),
                    user_prompt=f"【待翻译片段】\n{segment}",
                    temperature=0.0,
                    max_tokens=500,
                ).strip()
            except Exception:  # noqa: BLE001
                return ""
            if not self._is_acceptable_translation(translated, segment):
                return ""
            translated_segments.append(translated)

        return "\n".join(translated_segments).strip()

    def _split_translation_segments(self, content: str) -> tuple[str, ...]:
        normalized = " ".join(content.replace("\n", " ").split()).strip()
        if not normalized:
            return ()

        parts = re.split(r"(?<=[.!?;:])\s+", normalized)
        simple_parts = tuple(part.strip() for part in parts if part.strip())
        if len(simple_parts) > 1:
            return simple_parts

        segments: list[str] = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= 220:
                current = candidate
                continue
            if current:
                segments.append(current)
            current = part
        if current:
            segments.append(current)
        return tuple(segments)

    def _is_acceptable_translation(self, translated_text: str, source_text: str) -> bool:
        cleaned = translated_text.strip()
        if self._looks_like_empty_learning_text(cleaned):
            return False

        chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in cleaned)
        ascii_letters = sum(char.isascii() and char.isalpha() for char in cleaned)
        normalized_source = " ".join(source_text.split()).strip().lower()
        normalized_translated = " ".join(cleaned.split()).strip().lower()

        if normalized_source and normalized_translated == normalized_source:
            return False
        if chinese_chars >= 20:
            return True
        if chinese_chars >= 8 and chinese_chars >= ascii_letters:
            return True
        return not self._is_english_heavy(cleaned)

    def _looks_like_empty_learning_text(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return True
        punctuation_only = cleaned.strip(":：-·…；，。！？[]{}()（）")
        if not punctuation_only:
            return True
        chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in cleaned)
        ascii_letters = sum(char.isascii() and char.isalpha() for char in cleaned)
        if len(cleaned) <= 4:
            return True
        if chinese_chars == 0 and ascii_letters <= 4:
            return True
        return False

    def _is_english_heavy(self, text: str) -> bool:
        latin_chars = sum(char.isascii() and char.isalpha() for char in text)
        chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in text)
        return latin_chars > chinese_chars * 2

    def _try_parse_structured_text(self, raw_text: str) -> object:
        cleaned = raw_text.strip()
        if not cleaned or cleaned[0] not in "{[":
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(cleaned)
            except (ValueError, SyntaxError, TypeError):
                continue
        return None

    def _build_translation_fallback(self, paper_section: PaperSection) -> str:
        _ = paper_section
        return "当前模型这一轮没有稳定产出完整中文译文。"


class ReadingAgent:
    """我只负责核心要点和术语百科。"""

    def __init__(
        self,
        llm_client: LLMClient,
        translation_agent: TranslationAgent | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._translation_agent = translation_agent or TranslationAgent(llm_client)

    def run(
        self,
        paper_section: PaperSection,
        *,
        translation: str,
        event_handler: LearningEventHandler | None = None,
    ) -> LearningOutputs:
        clean_translation = self._translation_agent.reuse_or_translate(paper_section, translation)
        self._emit(
            event_handler,
            kind="observation",
            message="中文译文阶段完成。",
        )
        packet = self._generate_learning_packet(
            paper_section,
            translation=clean_translation,
        )
        if packet is not None:
            core_points, glossary = packet
            self._emit(
                event_handler,
                kind="action",
                message="summarize_and_build_glossary",
                action_input={"agent": "reading"},
            )
            self._emit(
                event_handler,
                kind="thought",
                message="我一次性整理核心要点和术语解释，减少不必要的往返调用。",
            )
            self._emit(
                event_handler,
                kind="observation",
                message="核心要点提炼完成。",
            )
            self._emit(
                event_handler,
                kind="observation",
                message="术语百科整理完成。",
            )
            return LearningOutputs(
                translation=clean_translation,
                core_points=core_points,
                glossary=glossary,
            )

        self._emit(
            event_handler,
            kind="action",
            message="summarize_key_points",
            action_input={"agent": "reading"},
        )
        self._emit(
            event_handler,
            kind="thought",
            message="我现在只提炼这一段最值得记住的三个学术观点。",
        )
        core_points = self._ensure_learning_core_points(
            paper_section,
            "",
            translation=clean_translation,
        )
        self._emit(
            event_handler,
            kind="observation",
            message="核心要点提炼完成。",
        )
        self._emit(
            event_handler,
            kind="action",
            message="build_glossary",
            action_input={"agent": "reading"},
        )
        self._emit(
            event_handler,
            kind="thought",
            message="接下来我补充这段里最关键的术语解释，保证阅读不跳步。",
        )
        glossary = self._ensure_learning_glossary(
            paper_section,
            "",
            translation=clean_translation,
            core_points=core_points,
        )
        self._emit(
            event_handler,
            kind="observation",
            message="术语百科整理完成。",
        )
        return LearningOutputs(
            translation=clean_translation,
            core_points=core_points,
            glossary=glossary,
        )

    def _generate_learning_packet(
        self,
        paper_section: PaperSection,
        *,
        translation: str,
    ) -> tuple[str, str] | None:
        try:
            payload = self._llm_client.generate_json(
                system_prompt=self._build_learning_packet_system_prompt(),
                user_prompt=(
                    f"【论文片段原文】\n{paper_section.content}\n\n【中文译文】\n{translation}"
                ),
                temperature=0.1,
                max_tokens=1300,
            )
        except Exception:  # noqa: BLE001
            payload = None
        if not isinstance(payload, dict):
            return None

        core_points = self._normalize_learning_packet_field(
            payload.get("core_points", payload.get("semantic_evidence", "")),
            paper_section=paper_section,
            kind="core_points",
        )
        glossary = self._normalize_learning_packet_field(
            payload.get("glossary", payload.get("research_supplement", "")),
            paper_section=paper_section,
            kind="glossary",
        )
        if core_points.count("- ") < 3 or glossary.count("- ") < 2:
            return None
        return core_points, glossary

    def _build_learning_packet_system_prompt(self) -> str:
        return (
            "你是科研论文阅读助手。"
            "请同时输出这段论文的核心要点和术语解释，减少多轮往返。"
            "只返回 JSON 对象，字段包括："
            "`core_points`（3 条中文要点列表），"
            "`glossary`（2-3 条中文术语解释列表）。"
        )

    def _normalize_learning_packet_field(
        self,
        value: object,
        *,
        paper_section: PaperSection,
        kind: str,
    ) -> str:
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if kind == "core_points" and len(items) >= 3:
                return self._format_bullets(items[:3])
            if kind == "glossary" and len(items) >= 2:
                return self._format_bullets(items[:3])
        text_value = str(value).strip()
        if kind == "core_points":
            return self.normalize_core_points_text(text_value, paper_section)
        return self.normalize_glossary_text(text_value, paper_section)

    def normalize_core_points_text(self, raw_text: str, paper_section: PaperSection) -> str:
        parsed = self._try_parse_structured_text(raw_text)
        if isinstance(parsed, dict):
            items = self._extract_core_points_from_mapping(parsed)
            if items:
                return self._format_bullets(items[:3])
            return self._build_core_points_fallback(paper_section)
        cleaned = raw_text.strip()
        if not cleaned or cleaned.startswith("{") or cleaned.startswith("["):
            return self._build_core_points_fallback(paper_section)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) >= 3 and all(line.startswith("-") for line in lines[:3]):
            return "\n".join(lines[:3])
        return self._build_core_points_fallback(paper_section)

    def normalize_glossary_text(self, raw_text: str, paper_section: PaperSection) -> str:
        parsed = self._try_parse_structured_text(raw_text)
        if isinstance(parsed, dict):
            items = self._extract_glossary_items_from_mapping(parsed)
            if items:
                return self._format_bullets(items[:3])
            return self._build_term_glossary(paper_section)
        cleaned = raw_text.strip()
        if not cleaned or cleaned.startswith("{") or cleaned.startswith("["):
            return self._build_term_glossary(paper_section)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            return "\n".join(lines[:3])
        return self._build_term_glossary(paper_section)

    def _ensure_learning_core_points(
        self,
        paper_section: PaperSection,
        text: str,
        *,
        translation: str,
    ) -> str:
        normalized = self.normalize_core_points_text(text, paper_section)
        if text.strip() and normalized.count("- ") >= 3:
            return normalized

        try:
            payload = self._llm_client.generate_json(
                system_prompt=build_learning_core_points_system_prompt(),
                user_prompt=f"【论文片段原文】\n{paper_section.content}\n\n【中文译文】\n{translation}",
                temperature=0.1,
                max_tokens=900,
            )
        except Exception:  # noqa: BLE001
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("semantic_evidence", "")
            if isinstance(candidate, list):
                items = [str(item).strip() for item in candidate if str(item).strip()]
                if len(items) >= 3:
                    return self._format_bullets(items[:3])
            repaired = self.normalize_core_points_text(str(candidate), paper_section)
            if repaired.count("- ") >= 3:
                return repaired

        return self._build_core_points_fallback(paper_section)

    def _ensure_learning_glossary(
        self,
        paper_section: PaperSection,
        text: str,
        *,
        translation: str,
        core_points: str,
    ) -> str:
        normalized = self.normalize_glossary_text(text, paper_section)
        if text.strip() and normalized.count("- ") >= 2:
            return normalized

        try:
            payload = self._llm_client.generate_json(
                system_prompt=build_learning_glossary_system_prompt(),
                user_prompt=(
                    f"【论文片段原文】\n{paper_section.content}\n\n"
                    f"【中文译文】\n{translation}\n\n"
                    f"【核心要点】\n{core_points}"
                ),
                temperature=0.1,
                max_tokens=900,
            )
        except Exception:  # noqa: BLE001
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("research_supplement", "")
            if isinstance(candidate, list):
                items = [str(item).strip() for item in candidate if str(item).strip()]
                if len(items) >= 2:
                    return self._format_bullets(items[:3])
            repaired = self.normalize_glossary_text(str(candidate), paper_section)
            if repaired.count("- ") >= 2:
                return repaired

        return self._build_term_glossary(paper_section)

    def _try_parse_structured_text(self, raw_text: str) -> object:
        cleaned = raw_text.strip()
        if not cleaned or cleaned[0] not in "{[":
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(cleaned)
            except (ValueError, SyntaxError, TypeError):
                continue
        return None

    def _extract_core_points_from_mapping(self, payload: dict) -> list[str]:
        core_items: list[str] = []
        for key in (
            "core_problem",
            "key_innovation",
            "theoretical_motivation",
            "technical_approach",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                core_items.append(value.strip())
            elif isinstance(value, list):
                core_items.extend(str(item).strip() for item in value if str(item).strip())
        return core_items

    def _extract_glossary_items_from_mapping(self, payload: dict) -> list[str]:
        glossary_items: list[str] = []
        related_concepts = payload.get("related_concepts")
        if isinstance(related_concepts, dict):
            for term, explanation in related_concepts.items():
                glossary_items.append(f"**{term}**：{explanation}")
        for key in ("historical_context", "significance"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                glossary_items.append(value.strip())
        return glossary_items

    def _format_bullets(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items if item)

    def _build_core_points_fallback(self, paper_section: PaperSection) -> str:
        title = paper_section.title.strip() or "当前片段"
        short_content = paper_section.content.replace("\n", " ").strip()
        if len(short_content) > 90:
            short_content = short_content[:90] + "..."
        return (
            f"- 这段内容首先在说明“{title}”对应的问题背景或任务目标。\n"
            f"- 作者真正想强调的是这段机制为什么重要：{short_content}\n"
            + (
                "- 阅读时最值得抓住的是：它解决了什么痛点、提出了什么关键思路，"
                "以及它和已有方法差在哪里。"
            )
        )

    def _build_term_glossary(self, paper_section: PaperSection) -> str:
        glossary_map = {
            "grounding": (
                "Grounding 指把自然语言、视觉观察和环境状态真正对上号，"
                "也就是让模型知道一句话具体对应哪一部分场景或动作。"
            ),
            "topological map": (
                "Topological Map 是拓扑地图，更关注地点之间怎么连通，"
                "而不是精确几何坐标，适合做导航决策。"
            ),
            "on-the-fly": "On-the-fly 可以理解成边走边算、现场生成，不是提前把所有结果都准备好。",
            "dual-scale": (
                "Dual-scale 指双尺度，通常是在全局和局部两个粒度上同时建模，"
                "让系统既看整体路线，也看眼前细节。"
            ),
            "cross-modal": "Cross-modal 指跨模态，把语言、视觉、地图等不同类型的信息放在一起理解。",
            "encoder": "Encoder 是编码器，负责把原始输入整理成后续模块更容易处理的表示。",
            "decoder": "Decoder 是解码器，负责根据已有表示一步步生成动作、预测结果或最终输出。",
            "attention": "Attention 是注意力机制，本质上是在一堆信息里挑出当前最该关注的部分。",
            "topological": "Topological 强调连通关系和结构关系，而不是精确距离。",
        }
        text = paper_section.combined_text.lower()
        matched = [
            f"- **{term.title()}**：{explanation}"
            for term, explanation in glossary_map.items()
            if term in text
        ]
        unique_matched = list(dict.fromkeys(matched))[:3]
        if len(unique_matched) >= 2:
            return "\n".join(unique_matched)

        for term in self._infer_term_candidates(paper_section):
            if len(unique_matched) >= 3:
                break
            unique_matched.append(
                f"- **{term}**：这是本段阅读时值得额外注意的术语，"
                "建议结合上下文理解它在任务设定或方法机制中的具体含义。"
            )
        return "\n".join(unique_matched[:3])

    def _infer_term_candidates(self, paper_section: PaperSection) -> tuple[str, ...]:
        text = paper_section.combined_text
        candidates = re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}(?:\s+[A-Z][A-Za-z0-9-]{2,})*\b", text)
        filtered: list[str] = []
        for item in candidates:
            if item.lower() in {"abstract", "introduction", "method", "conclusion"}:
                continue
            if item not in filtered:
                filtered.append(item)
        return tuple(filtered[:3])

    def _emit(self, handler: LearningEventHandler | None, **payload: object) -> None:
        if handler is not None:
            handler(payload)
