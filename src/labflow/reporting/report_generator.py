"""Markdown 报告生成器。"""

from __future__ import annotations

from dataclasses import dataclass

from labflow.reasoning.models import AlignmentResult


@dataclass(frozen=True)
class ReportSummary:
    """报告摘要。"""

    total_items: int
    high_risk_items: int
    good_items: int
    overall_confidence: float


class ReportGenerator:
    """我把结构化结果进一步压成审计周报，重点优化语义呈现和开发者体验。"""

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

        lines.extend(
            [
                "",
                "## 🔴 高风险错配项",
            ]
        )
        if high_risk_items:
            for index, result in enumerate(
                sorted(high_risk_items, key=lambda item: item.alignment_score),
                start=1,
            ):
                lines.extend(self._format_result_block(index, result, marker="🔴"))
        else:
            lines.append("- 当前没有识别出高风险错配项。")

        lines.extend(
            [
                "",
                "## 🟢 一致性良好项",
            ]
        )
        if good_items:
            for index, result in enumerate(
                sorted(good_items, key=lambda item: item.alignment_score, reverse=True),
                start=1,
            ):
                lines.extend(self._format_result_block(index, result, marker="🟢"))
        else:
            lines.append("- 当前还没有足够强的一致性样本。")

        lines.extend(
            [
                "",
                "## 改进建议",
            ]
        )
        if improvement_suggestions:
            lines.extend(f"- {suggestion}" for suggestion in improvement_suggestions)
        else:
            lines.append("- 当前没有额外改进建议。")

        return "\n".join(lines).strip() + "\n"

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
