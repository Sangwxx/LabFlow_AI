"""Markdown 报告生成器。"""

from __future__ import annotations

from dataclasses import dataclass

from labflow.clients.llm_client import LLMClient
from labflow.reasoning.models import AlignmentResult


@dataclass(frozen=True)
class ReportSummary:
    """报告摘要。"""

    total_items: int
    high_risk_items: int
    good_items: int
    overall_confidence: float


@dataclass(frozen=True)
class ReadingNoteEntry:
    """阅读笔记的单条输入。"""

    paper_section_title: str
    paper_section_content: str
    paper_section_page_number: int
    paper_section_order: int
    alignment_result: AlignmentResult


class ReportGenerator:
    """把结构化结果整理成可下载的 Markdown 报告。"""

    def generate_markdown(
        self,
        *,
        results: tuple[AlignmentResult, ...],
        report_title: str = "LabFlow AI 审计周报",
        project_overview: tuple[str, ...] = (),
    ) -> str:
        """把对齐结果汇总成标准 Markdown 报告。"""

        summary = self.build_summary(results)
        high_risk_items = tuple(result for result in results if result.is_high_risk)
        good_items = tuple(result for result in results if result.is_good_alignment)
        improvement_suggestions = self.collect_improvement_suggestions(results)

        lines: list[str] = [
            f"# {report_title}",
            "",
            "## 项目概况",
            f"- 对齐候选总数：{summary.total_items}",
            f"- 高风险错配项：{summary.high_risk_items}",
            f"- 一致性良好项：{summary.good_items}",
            f"- 总体对齐置信度：{summary.overall_confidence:.1f}/10",
        ]

        if project_overview:
            lines.extend(f"- {item}" for item in project_overview)

        lines.extend(["", "## 🔴 高风险错配项"])
        if high_risk_items:
            for index, result in enumerate(
                sorted(high_risk_items, key=lambda item: item.alignment_score),
                start=1,
            ):
                lines.extend(self._format_result_block(index, result, marker="🔴"))
        else:
            lines.append("- 当前没有识别出高风险错配项。")

        lines.extend(["", "## 🟢 一致性良好项"])
        if good_items:
            for index, result in enumerate(
                sorted(good_items, key=lambda item: item.alignment_score, reverse=True),
                start=1,
            ):
                lines.extend(self._format_result_block(index, result, marker="🟢"))
        else:
            lines.append("- 当前还没有足够强的一致性样本。")

        lines.extend(["", "## 改进建议"])
        if improvement_suggestions:
            lines.extend(f"- {suggestion}" for suggestion in improvement_suggestions)
        else:
            lines.append("- 当前没有额外改进建议。")

        return "\n".join(lines).strip() + "\n"

    def generate_literature_notes_markdown(
        self,
        *,
        entries: tuple[ReadingNoteEntry, ...],
        llm_client: LLMClient | None = None,
        report_title: str = "LabFlow 文献阅读笔记",
        project_overview: tuple[str, ...] = (),
    ) -> str:
        """把已读片段和对应代码整理成可下载的阅读笔记。"""

        if not entries:
            return self._build_literature_notes_fallback(
                entries=entries,
                report_title=report_title,
                project_overview=project_overview,
            )

        if llm_client is not None:
            try:
                markdown = llm_client.generate_text(
                    system_prompt=self._build_literature_notes_system_prompt(),
                    user_prompt=self._build_literature_notes_user_prompt(
                        entries=entries,
                        report_title=report_title,
                        project_overview=project_overview,
                    ),
                    temperature=0.1,
                    max_tokens=2400,
                ).strip()
            except Exception:  # noqa: BLE001
                markdown = ""
            if markdown:
                return markdown.rstrip() + "\n"

        return self._build_literature_notes_fallback(
            entries=entries,
            report_title=report_title,
            project_overview=project_overview,
        )

    def build_summary(self, results: tuple[AlignmentResult, ...]) -> ReportSummary:
        """汇总风险和置信度指标。"""

        total_items = len(results)
        high_risk_items = sum(1 for result in results if result.is_high_risk)
        good_items = sum(1 for result in results if result.is_good_alignment)
        overall_confidence = (
            sum(result.score_out_of_ten for result in results) / total_items if total_items else 0.0
        )
        return ReportSummary(
            total_items=total_items,
            high_risk_items=high_risk_items,
            good_items=good_items,
            overall_confidence=overall_confidence,
        )

    def collect_improvement_suggestions(
        self,
        results: tuple[AlignmentResult, ...],
    ) -> tuple[str, ...]:
        """对改进建议做去重整理。"""

        suggestions: list[str] = []
        for result in results:
            suggestion = result.improvement_suggestion.strip()
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)
        return tuple(suggestions)

    def _format_result_block(
        self,
        index: int,
        result: AlignmentResult,
        *,
        marker: str,
    ) -> list[str]:
        """把单条结果格式化成 Markdown 块。"""

        return [
            f"### {marker} {index}. {result.paper_section_title} ↔ {result.code_file_name}",
            f"- 对齐评分：{result.score_out_of_ten:.1f}/10",
            f"- 匹配类型：{result.match_type}",
            f"- 分析结论：{result.analysis}",
            f"- 改进建议：{result.improvement_suggestion}",
            "",
        ]

    def _build_literature_notes_system_prompt(self) -> str:
        return (
            "你是科研论文阅读笔记生成器。"
            "你会根据已读论文片段、对应代码和对齐结果生成一份能直接保存的 Markdown 笔记。"
            "要求："
            "1. 只输出 Markdown 正文，不要额外说明。"
            "2. 每个片段必须写清楚论文说了什么、代码做了什么、二者为什么对应。"
            "3. 不要空话，不要泛化成 '图结构相关实现' 这类标签。"
            "4. 结尾给出 3 条总体结论和 3 条后续阅读建议。"
        )

    def _build_literature_notes_user_prompt(
        self,
        *,
        entries: tuple[ReadingNoteEntry, ...],
        report_title: str,
        project_overview: tuple[str, ...],
    ) -> str:
        sections = [
            f"# {report_title}",
            "",
            "## 项目背景",
        ]
        if project_overview:
            sections.extend(f"- {item}" for item in project_overview)
        else:
            sections.append("- 当前项目已记录若干论文片段与对应实现。")

        sections.extend(["", "## 已读片段"])
        for index, entry in enumerate(entries, start=1):
            result = entry.alignment_result
            sections.extend(
                [
                    f"### 片段 {index}：{entry.paper_section_title}",
                    f"- 页码：P{entry.paper_section_page_number}",
                    f"- 论文片段：{entry.paper_section_content}",
                    (
                        f"- 对应代码：{result.code_file_name} "
                        f"L{result.code_start_line}-L{result.code_end_line}"
                    ),
                    f"- 代码解释：{result.analysis}",
                    f"- 对齐关系：{result.implementation_chain}",
                ]
            )
            if result.source_guide:
                guide_lines = ", ".join(
                    f"{item.symbol_name}({item.file_name} L{item.start_line}-L{item.end_line})"
                    for item in result.source_guide[:3]
                )
                sections.append(f"- 相关实现：{guide_lines}")
            sections.append("")

        sections.extend(
            [
                "## 输出要求",
                "- 结构清楚，适合直接下载成阅读笔记。",
                "- 把每个片段的机制关系写透，不要只复述结果。",
            ]
        )
        return "\n".join(sections).strip()

    def _build_literature_notes_fallback(
        self,
        *,
        entries: tuple[ReadingNoteEntry, ...],
        report_title: str,
        project_overview: tuple[str, ...],
    ) -> str:
        lines: list[str] = [
            f"# {report_title}",
            "",
            "## 项目背景",
        ]
        if project_overview:
            lines.extend(f"- {item}" for item in project_overview)
        else:
            lines.append("- 当前项目已记录若干论文片段与对应实现。")

        lines.extend(["", "## 已读片段"])
        for index, entry in enumerate(entries, start=1):
            result = entry.alignment_result
            lines.extend(
                [
                    f"### 片段 {index}：{entry.paper_section_title}",
                    f"- 页码：P{entry.paper_section_page_number}",
                    f"- 论文片段：{entry.paper_section_content}",
                    (
                        f"- 对应代码：{result.code_file_name} "
                        f"L{result.code_start_line}-L{result.code_end_line}"
                    ),
                    f"- 论文要点：{result.semantic_evidence or result.analysis}",
                    f"- 代码作用：{result.analysis or result.implementation_chain}",
                    "",
                ]
            )

        lines.extend(
            [
                "## 总体结论",
                "- 当前笔记由已读片段和对应代码自动整理而成，适合作为后续复盘底稿。",
                "- 如果需要进一步精炼，可以继续点开更多片段后重新生成。",
                "",
                "## 后续阅读建议",
                "- 继续补充同一章节附近的上下文片段。",
                "- 优先追踪 `源码导览` 中出现的相关实现模块。",
                "- 对于仍然模糊的对齐，继续核对定义链和调用链。",
            ]
        )
        return "\n".join(lines).strip() + "\n"
