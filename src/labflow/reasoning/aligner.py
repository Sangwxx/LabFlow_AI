"""论文与代码的对齐器兼容层。"""

from __future__ import annotations

from labflow.clients.llm_client import LLMClient
from labflow.parsers.git_repo_parser import GitRepoParseResult
from labflow.parsers.pdf_parser import PDFParseResult
from labflow.reasoning.agent_executor import PlanAndExecuteAgent
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import AlignmentResult, CodeEvidence, PaperSection


class PaperCodeAligner:
    """我保留历史入口，但底层已经切到按需触发的 ReAct Agent。"""

    def __init__(self, llm_client=None, evidence_builder: EvidenceBuilder | None = None) -> None:
        self._llm_client = llm_client or LLMClient()
        self._evidence_builder = evidence_builder or EvidenceBuilder()
        self._agent = PlanAndExecuteAgent(
            llm_client=self._llm_client,
            evidence_builder=self._evidence_builder,
        )

    def align(
        self,
        pdf_result: PDFParseResult,
        repo_result: GitRepoParseResult,
        top_k: int = 3,
    ) -> tuple[AlignmentResult, ...]:
        """从解析结果出发执行批量对齐。"""

        paper_sections = self._evidence_builder.build_paper_sections(pdf_result)
        code_evidences = self._evidence_builder.build_code_evidences(repo_result)
        project_structure = self._evidence_builder.build_project_structure(repo_result)
        return self.align_inputs(
            paper_sections=paper_sections,
            code_evidences=code_evidences,
            project_structure=project_structure,
            top_k=top_k,
        )

    def align_inputs(
        self,
        paper_sections: tuple[PaperSection, ...],
        code_evidences: tuple[CodeEvidence, ...],
        project_structure: str = "",
        top_k: int = 3,
    ) -> tuple[AlignmentResult, ...]:
        """从手工构造的输入出发执行批量对齐。"""

        results: list[AlignmentResult] = []
        for section in paper_sections[: max(top_k, len(paper_sections))]:
            result = self._agent.run(
                section,
                code_evidences,
                project_structure=project_structure
                or self._evidence_builder.build_project_structure_from_evidences(code_evidences),
            )
            if result is not None:
                results.append(result)

        results.sort(key=lambda item: item.alignment_score, reverse=True)
        return tuple(results)

    def align_section(
        self,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        *,
        project_structure: str = "",
    ) -> AlignmentResult | None:
        """为工作区右栏暴露单段语义理解入口。"""

        return self._agent.run(
            paper_section,
            code_evidences,
            project_structure=project_structure
            or self._evidence_builder.build_project_structure_from_evidences(code_evidences),
        )


def align(
    pdf_result: PDFParseResult,
    repo_result: GitRepoParseResult,
    llm_client=None,
    top_k: int = 3,
) -> tuple[AlignmentResult, ...]:
    """模块级批量对齐入口。"""

    return PaperCodeAligner(llm_client=llm_client).align(
        pdf_result=pdf_result,
        repo_result=repo_result,
        top_k=top_k,
    )


def align_inputs(
    paper_sections: tuple[PaperSection, ...],
    code_evidences: tuple[CodeEvidence, ...],
    llm_client=None,
    top_k: int = 3,
    project_structure: str = "",
) -> tuple[AlignmentResult, ...]:
    """为手工测试案例暴露一个直接入口。"""

    return PaperCodeAligner(llm_client=llm_client).align_inputs(
        paper_sections=paper_sections,
        code_evidences=code_evidences,
        project_structure=project_structure,
        top_k=top_k,
    )


def align_section(
    paper_section: PaperSection,
    code_evidences: tuple[CodeEvidence, ...],
    llm_client=None,
    top_k: int = 4,
    project_structure: str = "",
) -> AlignmentResult | None:
    """为工作区右栏暴露单段语义理解入口。"""

    _ = top_k
    return PaperCodeAligner(llm_client=llm_client).align_section(
        paper_section=paper_section,
        code_evidences=code_evidences,
        project_structure=project_structure,
    )
