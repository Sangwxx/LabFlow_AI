"""论文与代码的对齐器。"""

from __future__ import annotations

import json

from labflow.clients.llm_client import LLMClient
from labflow.parsers.git_repo_parser import GitRepoParseResult
from labflow.parsers.pdf_parser import PDFParseResult
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import (
    AlignmentCandidate,
    AlignmentResult,
    CodeEvidence,
    PaperSection,
)


class PaperCodeAligner:
    """我把 RAG 召回和最终判定拆成两段，先召回，再让模型做细判断。"""

    def __init__(self, llm_client=None, evidence_builder: EvidenceBuilder | None = None) -> None:
        self._llm_client = llm_client or LLMClient()
        self._evidence_builder = evidence_builder or EvidenceBuilder()

    def align(
        self,
        pdf_result: PDFParseResult,
        repo_result: GitRepoParseResult,
        top_k: int = 2,
    ) -> tuple[AlignmentResult, ...]:
        """从解析结果出发执行对齐。"""

        candidates = self._evidence_builder.build_alignment_candidates(
            pdf_result=pdf_result,
            repo_result=repo_result,
            top_k=top_k,
        )
        return self.align_candidates(candidates)

    def align_inputs(
        self,
        paper_sections: tuple[PaperSection, ...],
        code_evidences: tuple[CodeEvidence, ...],
        top_k: int = 2,
    ) -> tuple[AlignmentResult, ...]:
        """从手工构造的输入出发执行对齐。"""

        candidates = self._evidence_builder.build_alignment_candidates_from_inputs(
            paper_sections=paper_sections,
            code_evidences=code_evidences,
            top_k=top_k,
        )
        return self.align_candidates(candidates)

    def align_candidates(
        self,
        candidates: tuple[AlignmentCandidate, ...],
    ) -> tuple[AlignmentResult, ...]:
        """对候选对执行结构化推理。"""

        results: list[AlignmentResult] = []
        for candidate in candidates:
            payload = self._llm_client.generate_json(
                system_prompt=self._build_system_prompt(),
                user_prompt=self._build_user_prompt(candidate),
            )
            results.append(AlignmentResult.from_payload(payload, candidate))

        results.sort(key=lambda item: item.alignment_score, reverse=True)
        return tuple(results)

    def _build_system_prompt(self) -> str:
        """构造系统提示词。"""

        schema = {
            "alignment_score": "0 到 1 之间的小数",
            "match_type": (
                "strong_match | partial_match | missing_implementation | formula_mismatch"
            ),
            "analysis": "中文分析结论",
            "improvement_suggestion": "中文改进建议",
        }
        return (
            "你是一个只看证据说话的论文代码对齐审查助手。"
            "我已经通过 RAG 风格召回把最相关的章节和代码证据压缩给你，"
            "现在你只需要判断它们是否真的对得上。"
            "重点盯住两类问题："
            "一是论文里明确提出了方法、参数或公式，但代码没有实现；"
            "二是代码实现了相近逻辑，但参数、公式、损失项或约束与论文描述不一致。"
            "只输出 JSON，不要额外补充解释。"
            f"输出结构必须严格满足: {json.dumps(schema, ensure_ascii=False)}"
        )

    def _build_user_prompt(self, candidate: AlignmentCandidate) -> str:
        """构造用户提示词。"""

        commit_context = (
            "\n".join(candidate.code_evidence.commit_context)
            if candidate.code_evidence.commit_context
            else "无"
        )
        rules = "\n".join(
            [
                "1. 如果论文强调了关键参数、损失项、公式或约束，但代码片段里没有体现，"
                "请优先判为 missing_implementation。",
                "2. 如果代码片段出现了相同变量名，但数值、组合方式、符号方向或公式结构"
                "和论文不一致，请优先判为 formula_mismatch。",
                "3. 如果证据只覆盖了一部分实现，请判为 partial_match。",
                "4. 只有论文描述和代码实现明显一致时，才判为 strong_match。",
                "5. analysis 里必须点出你看到的证据，不要说空话。",
                "6. improvement_suggestion 里给出一条能落地的修复建议。",
            ]
        )
        return f"""
请分析下面这组论文章节和代码证据是否一致，并按约定 JSON 结构返回。

【召回得分】
{candidate.retrieval_score}

【论文章节标题】
{candidate.paper_section.title}

【论文章节内容】
{candidate.paper_section.content}

【代码文件】
{candidate.code_evidence.file_name}

【代码片段】
{candidate.code_evidence.code_snippet}

【关联 Git Diff】
{candidate.code_evidence.related_git_diff}

【最近提交上下文】
{commit_context}

判断要求：
{rules}
""".strip()


def align(
    pdf_result: PDFParseResult,
    repo_result: GitRepoParseResult,
    llm_client=None,
    top_k: int = 2,
) -> tuple[AlignmentResult, ...]:
    """模块级对齐入口。"""

    return PaperCodeAligner(llm_client=llm_client).align(
        pdf_result=pdf_result,
        repo_result=repo_result,
        top_k=top_k,
    )


def align_inputs(
    paper_sections: tuple[PaperSection, ...],
    code_evidences: tuple[CodeEvidence, ...],
    llm_client=None,
    top_k: int = 2,
) -> tuple[AlignmentResult, ...]:
    """为手工测试案例暴露一个直接入口。"""

    return PaperCodeAligner(llm_client=llm_client).align_inputs(
        paper_sections=paper_sections,
        code_evidences=code_evidences,
        top_k=top_k,
    )
