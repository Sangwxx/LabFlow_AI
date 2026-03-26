from __future__ import annotations

import ast
import re
from dataclasses import replace

from labflow.clients.llm_client import LLMClient
from labflow.reasoning.agent_engine import (
    AgentEventHandler,
    PlanAndExecuteEngine,
)
from labflow.reasoning.agent_prompts import (
    build_final_answer_system_prompt,
    build_final_answer_user_prompt,
    build_reflection_system_prompt,
    build_reflection_user_prompt,
)
from labflow.reasoning.code_knowledge_index import CodeKnowledgeIndex
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import (
    AlignmentCandidate,
    AlignmentResult,
    CodeEvidence,
    CodeSemanticSummary,
    ExecutionPlan,
    PaperSection,
    SourceGuideItem,
    StepExecutionTrace,
)


class CodeGroundingAgent:
    """我只负责论文片段与源码逻辑的对齐和解释。"""

    EXECUTION_BUDGET_SEC = 8.0

    def __init__(
        self,
        llm_client: LLMClient,
        evidence_builder: EvidenceBuilder,
        engine: PlanAndExecuteEngine,
    ) -> None:
        self._llm_client = llm_client
        self._evidence_builder = evidence_builder
        self._engine = engine

    def run(
        self,
        *,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        project_structure: str,
        role_prompt: str,
        plan: ExecutionPlan,
        event_handler: AgentEventHandler | None = None,
    ) -> AlignmentResult | None:
        semantic_index = self._evidence_builder.build_semantic_index_from_evidences(
            code_evidences,
            llm_client=self._llm_client,
        )
        self._emit_semantic_cards(semantic_index, event_handler)
        initial_candidates = self._build_initial_candidates(
            paper_section=paper_section,
            code_evidences=code_evidences,
            code_focus=plan.code_focus,
            semantic_index=semantic_index,
        )
        if not initial_candidates:
            self._emit(
                event_handler,
                kind="observation",
                message="??????????????????????????????",
            )
            return None

        current_candidates = initial_candidates
        if self._should_short_circuit_grounding(
            paper_section=paper_section,
            initial_candidates=initial_candidates,
            code_focus=plan.code_focus,
        ):
            self._emit(
                event_handler,
                kind="observation",
                message="???????????????????????????????",
            )
            reflected_result = self._build_local_fallback_result(
                paper_section,
                initial_candidates,
                (),
                plan,
            )
        else:
            execution_budget = self._resolve_execution_budget(
                paper_section,
                code_evidences=code_evidences,
            )
            current_plan, step_traces, current_candidates = self._engine.run(
                paper_section=paper_section,
                project_structure=project_structure,
                code_evidences=code_evidences,
                current_candidates=initial_candidates,
                role_prompt=role_prompt,
                event_handler=event_handler,
                max_runtime_sec=execution_budget,
            )
            current_candidates = self._stabilize_candidates(
                initial_candidates,
                current_candidates,
                paper_section=paper_section,
                code_focus=plan.code_focus,
            )
            if (
                all(not trace.tool_invocations for trace in step_traces)
                or current_plan.final_summary == "?????????????????????"
            ):
                self._emit(
                    event_handler,
                    kind="observation",
                    message="????????????????????????????????????",
                )
                reflected_result = self._build_local_fallback_result(
                    paper_section,
                    current_candidates,
                    step_traces,
                    current_plan,
                )
            else:
                result = self._build_final_answer(
                    paper_section=paper_section,
                    current_candidates=current_candidates,
                    step_traces=step_traces,
                    current_plan=current_plan,
                    role_prompt=role_prompt,
                )
                reflected_result = self._reflect(
                    result,
                    paper_section=paper_section,
                    role_prompt=role_prompt,
                )
        if self._should_refuse_alignment(
            paper_section,
            reflected_result,
            current_candidates,
        ):
            return None
        return self._enrich_result(
            reflected_result,
            paper_section=paper_section,
            current_candidates=current_candidates,
            semantic_index=semantic_index,
            project_structure=project_structure,
        )

    def _resolve_execution_budget(
        self,
        paper_section: PaperSection,
        *,
        code_evidences: tuple[CodeEvidence, ...],
    ) -> float:
        budget = self.EXECUTION_BUDGET_SEC
        paper_text = paper_section.combined_text.lower()
        if len(code_evidences) >= 300:
            budget = min(budget, 14.0)
        if self._is_topological_mapping_section(paper_section):
            budget = min(budget, 12.0)
        if any(term in paper_text for term in ("coarse-scale", "cross-modal", "encoder")):
            budget = min(budget, 9.0)
        return budget

    def _should_use_llm_rerank(
        self,
        paper_section: PaperSection,
        code_focus: tuple[str, ...],
    ) -> bool:
        if self._llm_client is None:
            return False
        if self._is_overview_section(paper_section):
            return False
        if self._is_topological_mapping_section(paper_section):
            return False
        paper_text = paper_section.combined_text.lower()
        return any(
            term in paper_text or term in " ".join(code_focus).lower()
            for term in ("coarse-scale", "cross-modal", "graph-aware", "encoder")
        )

    def _should_short_circuit_grounding(
        self,
        *,
        paper_section: PaperSection,
        initial_candidates: tuple[AlignmentCandidate, ...],
        code_focus: tuple[str, ...],
    ) -> bool:
        if not initial_candidates:
            return False
        effective_focus = code_focus or self._derive_focus_terms(paper_section)
        if self._is_overview_section(paper_section):
            return True
        if len(initial_candidates) < 2:
            return False
        top_score = self._candidate_priority_score(
            initial_candidates[0],
            paper_section=paper_section,
            code_focus=effective_focus,
        )
        second_score = self._candidate_priority_score(
            initial_candidates[1],
            paper_section=paper_section,
            code_focus=effective_focus,
        )
        if self._is_topological_mapping_section(paper_section):
            return top_score >= 12.0 and top_score - second_score >= 2.0
        return top_score >= 11.0 and top_score - second_score >= 3.0

    def _build_initial_candidates(
        self,
        *,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        code_focus: tuple[str, ...],
        semantic_index: tuple[CodeSemanticSummary, ...],
    ) -> tuple[AlignmentCandidate, ...]:
        merged: dict[str, AlignmentCandidate] = {}
        effective_focus = code_focus or self._derive_focus_terms(paper_section)
        knowledge_index = CodeKnowledgeIndex(
            semantic_index,
            llm_client=self._llm_client,
        )

        def collect(candidates: tuple[AlignmentCandidate, ...], score_boost: float = 0.0) -> None:
            for candidate in candidates:
                normalized_candidate = (
                    candidate
                    if score_boost == 0.0
                    else replace(
                        candidate,
                        retrieval_score=round(candidate.retrieval_score + score_boost, 4),
                    )
                )
                candidate_id = self._candidate_id(normalized_candidate)
                existing = merged.get(candidate_id)
                if (
                    existing is None
                    or normalized_candidate.retrieval_score > existing.retrieval_score
                ):
                    merged[candidate_id] = normalized_candidate

        def collect_for_section(
            section: PaperSection,
            *,
            lexical_boost: float = 0.0,
            semantic_boost: float = 0.0,
            knowledge_boost: float = 0.0,
            use_llm_rerank: bool = False,
        ) -> None:
            lexical_candidates = self._evidence_builder.build_alignment_candidates_from_inputs(
                paper_sections=(section,),
                code_evidences=code_evidences,
                top_k=12,
            )
            semantic_candidates = self._evidence_builder.retrieve_semantic_candidates(
                section,
                semantic_index,
                top_k=12,
            )
            knowledge_candidates = knowledge_index.search(
                section,
                focus_terms=effective_focus,
                top_k=10,
                use_llm_rerank=use_llm_rerank,
            )
            collect(lexical_candidates, lexical_boost)
            collect(semantic_candidates, semantic_boost)
            collect(knowledge_candidates, knowledge_boost)

        collect_for_section(
            paper_section,
            lexical_boost=0.02,
            semantic_boost=0.12,
            knowledge_boost=0.18,
            use_llm_rerank=self._should_use_llm_rerank(paper_section, effective_focus),
        )
        if not self._is_overview_section(
            paper_section
        ) and not self._is_topological_mapping_section(paper_section):
            for focus in effective_focus[:2]:
                synthetic_section = PaperSection(
                    title=paper_section.title,
                    content=focus,
                    level=paper_section.level,
                    page_number=paper_section.page_number,
                    order=paper_section.order,
                    block_orders=paper_section.block_orders,
                )
                collect_for_section(
                    synthetic_section,
                    lexical_boost=0.0,
                    semantic_boost=0.09,
                    knowledge_boost=0.12,
                    use_llm_rerank=False,
                )

        traced_symbols = self._extract_trace_symbols(effective_focus, semantic_index, merged)
        if traced_symbols:
            traced_candidates = self._evidence_builder.trace_related_candidates(
                paper_section,
                semantic_index,
                trace_symbols=traced_symbols,
                seen_candidate_ids=set(merged.keys()),
                limit=4,
            )
            collect(traced_candidates, 0.08)

        return tuple(
            sorted(
                merged.values(),
                key=lambda item: self._candidate_sort_key(
                    item,
                    paper_section=paper_section,
                    code_focus=effective_focus,
                ),
                reverse=True,
            )[:6]
        )

    def _extract_trace_symbols(
        self,
        code_focus: tuple[str, ...],
        semantic_index: tuple[CodeSemanticSummary, ...],
        merged_candidates: dict[str, AlignmentCandidate],
    ) -> tuple[str, ...]:
        trace_symbols: list[str] = []
        for focus in code_focus:
            normalized = focus.strip()
            if normalized and normalized not in trace_symbols:
                trace_symbols.append(normalized)

        top_candidate_ids = set(list(merged_candidates.keys())[:3])
        for summary in semantic_index:
            if summary.identity not in top_candidate_ids:
                continue
            for symbol in (
                *summary.defined_symbols,
                *summary.called_symbols,
                *summary.anchor_terms,
            ):
                if len(symbol) < 3:
                    continue
                if symbol not in trace_symbols:
                    trace_symbols.append(symbol)
        return tuple(trace_symbols[:8])

    def _emit_semantic_cards(
        self,
        semantic_index: tuple[CodeSemanticSummary, ...],
        event_handler: AgentEventHandler | None,
    ) -> None:
        if not semantic_index:
            return
        top_cards = semantic_index[:4]
        lines = []
        for summary in top_cards:
            evidence = summary.code_evidence
            lines.append(
                f"{evidence.file_name} | "
                f"L{evidence.start_line}-L{evidence.end_line} | "
                f"{summary.summary}"
            )
        self._emit(
            event_handler,
            kind="observation",
            message="候选代码卡片：\n" + "\n".join(lines),
        )

    def _enrich_result(
        self,
        result: AlignmentResult,
        *,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        semantic_index: tuple[CodeSemanticSummary, ...],
        project_structure: str,
    ) -> AlignmentResult:
        ordered_candidates = self._order_candidates_for_guide(result, current_candidates)
        source_guide = self._build_source_guide(
            paper_section=paper_section,
            current_candidates=ordered_candidates,
            semantic_index=semantic_index,
        )
        project_structure_context = self._build_project_structure_context(
            project_structure,
            ordered_candidates,
        )
        return replace(
            result,
            source_guide=source_guide,
            project_structure_context=project_structure_context,
        )

    def _order_candidates_for_guide(
        self,
        result: AlignmentResult,
        current_candidates: tuple[AlignmentCandidate, ...],
    ) -> tuple[AlignmentCandidate, ...]:
        selected_id = f"{result.code_file_name}:{result.code_start_line}-{result.code_end_line}"
        prioritized: list[AlignmentCandidate] = []
        seen_ids: set[str] = set()
        for candidate in current_candidates:
            candidate_id = self._candidate_id(candidate)
            if candidate_id == selected_id:
                prioritized.append(candidate)
                seen_ids.add(candidate_id)
                break
        for candidate in current_candidates:
            candidate_id = self._candidate_id(candidate)
            if candidate_id in seen_ids:
                continue
            prioritized.append(candidate)
            seen_ids.add(candidate_id)
        return tuple(prioritized)

    def _build_source_guide(
        self,
        *,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        semantic_index: tuple[CodeSemanticSummary, ...],
    ) -> tuple[SourceGuideItem, ...]:
        guide_candidates = self._select_guide_candidates(
            paper_section=paper_section,
            current_candidates=current_candidates,
            semantic_index=semantic_index,
        )
        summary_lookup = {summary.identity: summary for summary in semantic_index}
        guide_items: list[SourceGuideItem] = []
        for candidate in guide_candidates[:4]:
            evidence = candidate.code_evidence
            summary = summary_lookup.get(self._candidate_id(candidate))
            summary_text = self._build_guide_summary(
                evidence,
                semantic_summary=(summary.summary if summary is not None else ""),
            )
            responsibilities = (
                tuple(item for item in summary.responsibilities[:3] if item.strip())
                if summary is not None and summary.responsibilities
                else self._build_guide_responsibilities(evidence)
            )
            guide_items.append(
                SourceGuideItem(
                    file_name=evidence.file_name,
                    symbol_name=evidence.symbol_name or evidence.parent_symbol or "未命名逻辑块",
                    block_type=evidence.block_type,
                    start_line=evidence.start_line,
                    end_line=evidence.end_line,
                    summary=summary_text,
                    responsibilities=responsibilities,
                    relevance_reason=self._build_guide_relevance_reason(paper_section, evidence),
                    code_preview=self._build_code_preview(evidence),
                    retrieval_score=round(candidate.retrieval_score, 4),
                )
            )
        return tuple(guide_items)

    def _select_guide_candidates(
        self,
        *,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        semantic_index: tuple[CodeSemanticSummary, ...],
    ) -> tuple[AlignmentCandidate, ...]:
        code_focus = self._derive_focus_terms(paper_section)
        ordered_candidates = tuple(
            sorted(
                current_candidates,
                key=lambda item: self._candidate_sort_key(
                    item,
                    paper_section=paper_section,
                    code_focus=code_focus,
                ),
                reverse=True,
            )
        )
        promoted_candidates: list[AlignmentCandidate] = []
        seen_ids: set[str] = set()
        for candidate in ordered_candidates:
            promoted_candidate = self._promote_candidate_for_guide(
                candidate,
                paper_section=paper_section,
                semantic_index=semantic_index,
                code_focus=code_focus,
            )
            promoted_id = self._candidate_id(promoted_candidate)
            if promoted_id in seen_ids:
                continue
            if self._is_trivial_helper_evidence(promoted_candidate.code_evidence):
                continue
            promoted_candidates.append(promoted_candidate)
            seen_ids.add(promoted_id)
        if promoted_candidates:
            return tuple(promoted_candidates)
        return ordered_candidates

    def _promote_candidate_for_guide(
        self,
        candidate: AlignmentCandidate,
        *,
        paper_section: PaperSection,
        semantic_index: tuple[CodeSemanticSummary, ...],
        code_focus: tuple[str, ...],
    ) -> AlignmentCandidate:
        representative = self._resolve_representative_evidence(
            candidate.code_evidence,
            paper_section=paper_section,
            semantic_index=semantic_index,
            code_focus=code_focus,
        )
        if representative == candidate.code_evidence:
            return candidate
        return AlignmentCandidate(
            paper_section=candidate.paper_section,
            code_evidence=representative,
            retrieval_score=candidate.retrieval_score,
        )

    def _resolve_representative_evidence(
        self,
        evidence: CodeEvidence,
        *,
        paper_section: PaperSection,
        semantic_index: tuple[CodeSemanticSummary, ...],
        code_focus: tuple[str, ...],
    ) -> CodeEvidence:
        symbol_name = (evidence.symbol_name or "").lower()
        if (
            evidence.block_type in {"method", "function"}
            and not self._is_trivial_helper_evidence(evidence)
            and not symbol_name.endswith(".__init__")
        ):
            return evidence

        related_evidences = [evidence]
        for summary in semantic_index:
            candidate = summary.code_evidence
            if candidate.file_name != evidence.file_name:
                continue
            if not self._belongs_to_same_guide_cluster(evidence, candidate):
                continue
            if candidate not in related_evidences:
                related_evidences.append(candidate)
        if self._is_trivial_helper_evidence(evidence):
            for summary in semantic_index:
                candidate = summary.code_evidence
                if candidate.file_name != evidence.file_name:
                    continue
                if self._is_trivial_helper_evidence(candidate):
                    continue
                if candidate not in related_evidences:
                    related_evidences.append(candidate)
        best_evidence = max(
            related_evidences,
            key=lambda item: self._guide_representation_score(
                item,
                paper_section=paper_section,
                code_focus=code_focus,
            ),
        )
        if best_evidence.block_type == "class" and best_evidence.symbol_name:
            child_candidates = [
                candidate
                for candidate in related_evidences
                if candidate.parent_symbol == best_evidence.symbol_name
                and candidate.block_type in {"method", "function"}
                and not self._is_trivial_helper_evidence(candidate)
            ]
            non_init_children = [
                candidate
                for candidate in child_candidates
                if not (candidate.symbol_name or "").lower().endswith(".__init__")
            ]
            if non_init_children:
                child_candidates = non_init_children
            if child_candidates:
                return max(
                    child_candidates,
                    key=lambda item: self._guide_representation_score(
                        item,
                        paper_section=paper_section,
                        code_focus=code_focus,
                    ),
                )
        return best_evidence

    def _belongs_to_same_guide_cluster(
        self,
        anchor: CodeEvidence,
        candidate: CodeEvidence,
    ) -> bool:
        if anchor.file_name != candidate.file_name:
            return False
        if anchor == candidate:
            return True

        anchor_symbol = anchor.symbol_name or ""
        anchor_parent = anchor.parent_symbol or ""
        candidate_symbol = candidate.symbol_name or ""
        candidate_parent = candidate.parent_symbol or ""

        if anchor_parent:
            return candidate_symbol == anchor_parent or candidate_parent == anchor_parent
        if anchor.block_type == "class" and anchor_symbol:
            return candidate_parent == anchor_symbol or candidate_symbol == anchor_symbol
        if "." in anchor_symbol:
            owner = anchor_symbol.split(".", 1)[0]
            return candidate_parent == owner or candidate_symbol == owner
        return False

    def _guide_representation_score(
        self,
        evidence: CodeEvidence,
        *,
        paper_section: PaperSection,
        code_focus: tuple[str, ...],
    ) -> float:
        span = max(1, evidence.end_line - evidence.start_line + 1)
        score = (
            self._compute_focus_alignment_score(
                evidence,
                paper_section=paper_section,
                code_focus=code_focus,
            )
            * 2.0
            + self._compute_mechanism_alignment_score(
                evidence,
                paper_section=paper_section,
            )
            * 3.0
            + self._compute_specificity_score(
                evidence,
                paper_section=paper_section,
            )
        )
        if evidence.block_type in {"method", "function"} and span >= 6:
            score += 1.2
        if evidence.block_type == "class":
            score += 0.6
        if evidence.block_type == "module_intro":
            score -= 8.0
        if (evidence.symbol_name or "").lower().endswith(".__init__"):
            score -= 3.2
        if evidence.file_name.lower().endswith("/env.py") or evidence.file_name.lower().endswith(
            "\\env.py"
        ):
            score -= 4.0
        if self._is_topological_mapping_section(paper_section):
            lowered_symbol = (evidence.symbol_name or "").lower()
            lowered_file = evidence.file_name.lower()
            if any(
                token in lowered_symbol
                for token in ("teacher_action", "make_equiv_action", "rollout")
            ):
                score -= 10.0
            if any(
                token in lowered_file
                for token in (
                    "/r2r/agent.py",
                    "\\r2r\\agent.py",
                    "/reverie/agent",
                    "\\reverie\\agent",
                )
            ):
                score -= 6.0
            if any(token in lowered_file for token in ("graph_utils.py", "vilmodel.py")):
                score += 2.8
            if any(
                token in lowered_symbol
                for token in ("update_graph", "get_pos_fts", "floydgraph.path", "graphmap")
            ):
                score += 3.2
        if span <= 3:
            score -= 4.0
        if self._is_trivial_helper_evidence(evidence):
            score -= 6.0
        return score

    def _build_project_structure_context(
        self,
        project_structure: str,
        current_candidates: tuple[AlignmentCandidate, ...],
    ) -> tuple[str, ...]:
        normalized_paths = [
            line.replace(" / ", "/").strip()
            for line in project_structure.splitlines()
            if line.strip()
        ]
        if not normalized_paths:
            return ()

        top_level_counts: dict[str, int] = {}
        directory_counts: dict[str, int] = {}
        for path in normalized_paths:
            parts = [part for part in path.split("/") if part]
            if not parts:
                continue
            top_level_counts[parts[0]] = top_level_counts.get(parts[0], 0) + 1
            if len(parts) >= 2:
                directory_key = "/".join(parts[:2])
            else:
                directory_key = parts[0]
            directory_counts[directory_key] = directory_counts.get(directory_key, 0) + 1

        top_level_summary = " / ".join(
            f"{name}（{count}）"
            for name, count in sorted(
                top_level_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:4]
        )
        context_lines = [f"项目主干目录：{top_level_summary}。"] if top_level_summary else []

        seen_directories: set[str] = set()
        for candidate in current_candidates[:3]:
            directory_key = self._extract_directory_key(candidate.code_evidence.file_name)
            if not directory_key or directory_key in seen_directories:
                continue
            seen_directories.add(directory_key)
            file_count = directory_counts.get(directory_key, 1)
            directory_role = self._describe_directory_role(directory_key)
            context_lines.append(
                f"{directory_key}：{directory_role}，当前目录共 {file_count} 个文件。"
            )
        return tuple(context_lines)

    def _extract_directory_key(self, file_name: str) -> str:
        parts = [part for part in file_name.split("/") if part]
        if not parts:
            return ""
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return parts[0]

    def _describe_directory_role(self, directory_key: str) -> str:
        normalized = directory_key.lower()
        if "models" in normalized:
            return "承载模型主干、编码层和核心机制实现"
        if any(marker in normalized for marker in ("graph", "utils", "tool")):
            return "承载图结构维护、工具函数与底层支撑逻辑"
        if any(marker in normalized for marker in ("agent", "reverie", "r2r")):
            return "承载任务级 Agent、动作决策与环境交互逻辑"
        if "pretrain" in normalized:
            return "承载预训练阶段的编码模块与共享组件"
        if any(marker in normalized for marker in ("data", "dataset")):
            return "承载数据读取、组织与预处理逻辑"
        return "承载该方向下的相关实现"

    def _build_guide_summary(self, evidence: CodeEvidence, *, semantic_summary: str = "") -> str:
        code_text = evidence.code_snippet.lower()
        symbol_text = (evidence.symbol_name or evidence.parent_symbol or "").lower()
        if all(marker in code_text for marker in ("rel_angles", "rel_dists", "get_angle_fts")):
            return (
                "这段代码会遍历地图中的候选节点，计算相对朝向、距离和步数特征，"
                "并把它们拼成位置特征向量返回。"
            )
        if "add_edge" in code_text and "node_positions" in code_text:
            return (
                "这段代码会把当前观测写回地图，更新节点位置，"
                "并把当前节点与候选节点之间的边加入图结构。"
            )
        if "graph_sprels" in code_text and any(
            marker in code_text for marker in ("visual_attention", "visn_self_att", "ctx_att_mask")
        ):
            return (
                "这段代码负责这一层的前向传播：先让视觉特征和文本对齐，"
                "再把图结构偏置并入注意力计算，输出更新后的跨模态表示。"
            )
        if "modulelist" in code_text and "graphlxrtxlayer" in code_text:
            return (
                "这个模块把多层跨模态编码层串起来，负责反复更新图节点表示与文本表示，"
                "是编码器骨架的一部分。"
            )
        if "forward_navigation_per_step" in symbol_text:
            return (
                "这段代码负责单步导航决策：先分别算出全局地图分支和局部视角分支的表示，"
                "再把两路动作分数融合成最终导航 logits，"
                "并额外输出目标物体 grounding 所需的预测结果。"
            )
        if "gmap_input_embedding" in symbol_text:
            return (
                "这段代码负责生成全局地图节点的输入表示：先聚合同一节点的视觉特征，"
                "再叠加导航步编码和位置编码，并生成后续图编码器要用的掩码。"
            )
        structured_summary = self._build_structured_behavior_summary(evidence)
        if structured_summary:
            return structured_summary
        if evidence.docstring.strip():
            return f"这段代码主要负责：{evidence.docstring.strip()}"
        cleaned_summary = semantic_summary.strip()
        if cleaned_summary:
            return self._rewrite_semantic_summary_as_explanation(cleaned_summary, evidence)
        symbol = evidence.symbol_name or evidence.parent_symbol or evidence.block_type
        return f"这段代码主要围绕 `{symbol}` 展开，负责当前段落对应的这一段实现逻辑。"

    def _build_structured_behavior_summary(self, evidence: CodeEvidence) -> str:
        function_node = self._parse_python_function(evidence)
        if function_node is None:
            return ""

        if function_node.name == "__init__":
            return self._summarize_init_method(function_node, evidence)
        if self._looks_like_path_reconstruction(function_node, evidence):
            return self._summarize_path_reconstruction()
        if self._looks_like_serialization(function_node, evidence):
            return self._summarize_serialization(function_node)
        return self._summarize_function_flow(function_node)

    def _parse_python_function(
        self,
        evidence: CodeEvidence,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        snippet = evidence.code_snippet.strip()
        if not snippet.startswith(("def ", "async def ")):
            return None
        try:
            module = ast.parse(snippet)
        except SyntaxError:
            return None
        if not module.body:
            return None
        first_node = module.body[0]
        if isinstance(first_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return first_node
        return None

    def _summarize_init_method(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
        evidence: CodeEvidence,
    ) -> str:
        attributes = self._extract_self_attributes(function_node)
        display_attrs = (
            "、".join(f"`{name}`" for name in attributes[:4]) if attributes else "类成员状态"
        )
        if any(
            marker in attributes
            for marker in ("graph", "node_positions", "node_embeds", "node_nav_scores")
        ):
            purpose = "后续写入观测、更新图边与维护节点状态做准备"
        else:
            purpose = "后续调用和状态更新做准备"
        owner = evidence.parent_symbol or "当前类"
        return (
            f"这段代码在初始化 `{owner}` 的运行时状态，建立 {display_attrs} 等成员，为{purpose}。"
        )

    def _looks_like_path_reconstruction(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
        evidence: CodeEvidence,
    ) -> bool:
        code_text = evidence.code_snippet.lower()
        return function_node.name == "path" and code_text.count("self.path(") >= 2

    def _summarize_path_reconstruction(self) -> str:
        return (
            "这段代码会根据图里记录的中间节点递归还原从 x 到 y 的路径："
            "如果两点直接相连就直接返回终点，否则把路径拆成两段再拼接起来。"
        )

    def _looks_like_serialization(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
        evidence: CodeEvidence,
    ) -> bool:
        lowered_name = function_node.name.lower()
        code_text = evidence.code_snippet.lower()
        return (
            any(marker in lowered_name for marker in ("save", "dump", "serialize"))
            or "json" in lowered_name
            or "json" in code_text
        )

    def _summarize_serialization(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str:
        attributes = self._extract_self_attributes(
            function_node
        ) + self._extract_referenced_self_attributes(function_node)
        interesting_attrs = [
            name
            for name in attributes
            if any(flag in name for flag in ("node", "edge", "graph", "score", "visited"))
        ]
        if interesting_attrs:
            display_attrs = "、".join(f"`{name}`" for name in interesting_attrs[:4])
            return (
                f"这段代码会把 {display_attrs} 等图状态整理成可保存的结构，"
                "方便导出、调试或复现当前运行结果。"
            )
        return "这段代码会把当前对象的关键信息整理成可保存的结构，方便后续导出、调试或复现。"

    def _summarize_function_flow(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str:
        loop_targets = self._extract_loop_targets(function_node)
        called_symbols = self._extract_called_symbols_from_ast(function_node)
        return_names = self._extract_return_names(function_node)

        parts: list[str] = []
        if loop_targets:
            display_targets = "、".join(f"`{name}`" for name in loop_targets[:2])
            parts.append(f"遍历 {display_targets}")
        if called_symbols:
            display_calls = "、".join(f"`{name}`" for name in called_symbols[:3])
            parts.append(f"调用 {display_calls}")
        if return_names:
            display_returns = "、".join(f"`{name}`" for name in return_names[:2])
            parts.append(f"最后返回 {display_returns}")

        if not parts:
            return ""
        return "这段代码会先" + "，再".join(parts[:3]) + "。"

    def _rewrite_semantic_summary_as_explanation(
        self,
        semantic_summary: str,
        evidence: CodeEvidence,
    ) -> str:
        lowered = semantic_summary.lower()
        if "定义了" in semantic_summary or "调用了" in semantic_summary:
            called_symbols = ", ".join(self._extract_called_symbols(evidence)[:4])
            if called_symbols:
                symbol = evidence.symbol_name or evidence.parent_symbol or evidence.block_type
                return (
                    f"这段代码主要围绕 `{symbol}` 展开，"
                    f"会调用 {called_symbols} 等逻辑来完成当前这一步。"
                )
        if lowered.startswith(evidence.file_name.lower()):
            symbol = evidence.symbol_name or evidence.parent_symbol or evidence.block_type
            return f"这段代码位于 `{evidence.file_name}`，主要承担 `{symbol}` 这一层逻辑。"
        return semantic_summary

    def _extract_called_symbols(self, evidence: CodeEvidence) -> tuple[str, ...]:
        called_symbols: list[str] = []
        for line in evidence.code_snippet.splitlines():
            stripped = line.strip()
            if "(" not in stripped or stripped.startswith(("def ", "class ", "#")):
                continue
            candidate = stripped.split("(", 1)[0].split()[-1].strip()
            candidate = candidate.split(".")[-1]
            if len(candidate) < 2:
                continue
            if candidate not in called_symbols:
                called_symbols.append(candidate)
        return tuple(called_symbols)

    def _extract_self_attributes(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, ...]:
        attributes: list[str] = []
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr not in attributes
                ):
                    attributes.append(target.attr)
        return tuple(attributes)

    def _extract_loop_targets(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, ...]:
        targets: list[str] = []
        for node in ast.walk(function_node):
            if isinstance(node, ast.For):
                target_name = self._format_name(node.iter)
                if target_name and target_name not in targets:
                    targets.append(target_name)
        return tuple(targets)

    def _extract_referenced_self_attributes(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, ...]:
        attributes: list[str] = []
        for node in ast.walk(function_node):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr not in attributes
            ):
                attributes.append(node.attr)
        return tuple(attributes)

    def _extract_called_symbols_from_ast(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, ...]:
        called_symbols: list[str] = []
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Call):
                continue
            name = self._format_name(node.func)
            if not name:
                continue
            symbol = name.split(".")[-1]
            if symbol not in called_symbols:
                called_symbols.append(symbol)
        return tuple(called_symbols)

    def _extract_return_names(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, ...]:
        return_names: list[str] = []
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            for child in ast.walk(node.value):
                if isinstance(child, ast.Name) and child.id not in return_names:
                    return_names.append(child.id)
        return tuple(return_names)

    def _format_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._format_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return self._format_name(node.func)
        if isinstance(node, ast.Subscript):
            return self._format_name(node.value)
        return ""

    def _build_guide_responsibilities(self, evidence: CodeEvidence) -> tuple[str, ...]:
        responsibilities: list[str] = []
        if evidence.parent_symbol:
            responsibilities.append(f"挂在 `{evidence.parent_symbol}` 这一层级下。")
        if evidence.docstring.strip():
            responsibilities.append(evidence.docstring.strip())
        elif evidence.symbol_name:
            responsibilities.append(f"核心符号是 `{evidence.symbol_name}`。")
        responsibilities.append(
            f"代码范围位于 L{evidence.start_line}-L{evidence.end_line}，适合继续沿调用链复核。"
        )
        return tuple(responsibilities[:3])

    def _build_guide_relevance_reason(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> str:
        return self._build_module_match_reason(paper_section, evidence)

    def _build_code_preview(self, evidence: CodeEvidence) -> str:
        preview_lines = [
            line.rstrip() for line in evidence.code_snippet.splitlines() if line.strip()
        ]
        return "\n".join(preview_lines[:8])

    def _build_final_answer(
        self,
        *,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        step_traces: tuple[StepExecutionTrace, ...],
        current_plan: ExecutionPlan,
        role_prompt: str,
    ) -> AlignmentResult:
        payload = self._llm_client.generate_json(
            system_prompt=build_final_answer_system_prompt(role_prompt),
            user_prompt=build_final_answer_user_prompt(
                paper_section,
                current_candidates,
                step_traces,
            ),
            temperature=0.1,
            max_tokens=2000,
        )
        if not isinstance(payload, dict):
            return self._build_local_fallback_result(
                paper_section,
                current_candidates,
                step_traces,
                current_plan,
            )

        try:
            selected_index = int(payload.get("best_candidate_index", 0))
        except (TypeError, ValueError):
            selected_index = 0
        selected_index = max(0, min(selected_index, len(current_candidates) - 1))

        result = AlignmentResult.from_payload(payload, current_candidates[selected_index])
        plan_steps = current_plan.steps or tuple(trace.step for trace in step_traces)
        evidence = current_candidates[selected_index].code_evidence
        implementation_chain = (
            self._build_grounded_implementation_chain(
                paper_section,
                evidence,
                result.implementation_chain,
                current_candidates,
            )
            if self._has_source_grounding(result, evidence)
            else ""
        )
        return replace(
            result,
            retrieval_plan="\n".join(step.display_text for step in plan_steps),
            plan_steps=plan_steps,
            step_traces=step_traces,
            agent_observations=tuple(trace.observation for trace in step_traces),
            operator_alignment=(
                result.operator_alignment or self._build_operator_alignment(paper_section, evidence)
            ),
            shape_alignment=(
                result.shape_alignment or self._build_shape_alignment(paper_section, evidence)
            ),
            implementation_chain=implementation_chain,
        )

    def _reflect(
        self,
        result: AlignmentResult,
        *,
        paper_section: PaperSection,
        role_prompt: str,
    ) -> AlignmentResult:
        payload = self._llm_client.generate_json(
            system_prompt=build_reflection_system_prompt(role_prompt),
            user_prompt=build_reflection_user_prompt(paper_section, result),
            temperature=0.0,
            max_tokens=900,
        )
        if not isinstance(payload, dict):
            payload = {
                "reflection": "当前没有稳定拿到反思结果，建议人工核对。",
                "final_confidence": result.alignment_score,
                "confidence_note": "模型反思阶段未稳定返回，建议人工复核。",
                "needs_manual_review": True,
            }

        try:
            reflected_score = float(payload.get("final_confidence", result.alignment_score))
        except (TypeError, ValueError):
            reflected_score = result.alignment_score
        reflected_score = max(0.0, min(1.0, reflected_score))

        confidence_note = str(payload.get("confidence_note", "")).strip()
        needs_manual_review = bool(payload.get("needs_manual_review", False))
        if reflected_score < 0.78 and not confidence_note:
            confidence_note = "当前源码证据还不够扎实，建议你人工复核关键实现行。"

        return replace(
            result,
            alignment_score=reflected_score,
            reflection=str(payload.get("reflection", "当前缺少自我审计结论。")).strip(),
            confidence_note=confidence_note,
            needs_manual_review=needs_manual_review or reflected_score < 0.78,
        )

    def _should_refuse_alignment(
        self,
        paper_section: PaperSection,
        result: AlignmentResult,
        current_candidates: tuple[AlignmentCandidate, ...],
    ) -> bool:
        if not current_candidates:
            return True
        if result.match_type == "strong_match" and bool(result.implementation_chain.strip()):
            return False
        if bool(result.implementation_chain.strip()) and result.alignment_score >= 0.58:
            if self._has_direct_semantic_anchor(paper_section, current_candidates[0].code_evidence):
                return False
            if self._looks_like_infrastructure_code(current_candidates[0].code_evidence):
                return True
            if current_candidates[0].retrieval_score >= 0.2:
                return False
        if self._has_direct_semantic_anchor(paper_section, current_candidates[0].code_evidence):
            return False
        if self._looks_like_infrastructure_code(current_candidates[0].code_evidence):
            return True
        return current_candidates[0].retrieval_score < 0.025 and result.alignment_score < 0.6

    def _has_direct_semantic_anchor(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> bool:
        paper_text = paper_section.combined_text
        code_text = evidence.combined_text.lower()
        anchors = (
            "attention",
            "softmax",
            "encoder",
            "decoder",
            "fusion",
            "graph",
            "loss",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "optimizer",
            "momentum",
            "map",
            "policy",
        )
        return any(anchor in paper_text and anchor in code_text for anchor in anchors)

    def _looks_like_infrastructure_code(self, evidence: CodeEvidence) -> bool:
        file_name = evidence.file_name.lower()
        infrastructure_markers = (
            "utils/",
            "util/",
            "logger",
            "logging",
            "io.py",
            "config",
            "yaml",
            "json",
        )
        return any(marker in file_name for marker in infrastructure_markers)

    def _build_local_fallback_result(
        self,
        paper_section: PaperSection,
        current_candidates: tuple[AlignmentCandidate, ...],
        step_traces: tuple[StepExecutionTrace, ...],
        current_plan: ExecutionPlan,
    ) -> AlignmentResult:
        code_focus = self._derive_focus_terms(paper_section)
        candidate = sorted(
            current_candidates,
            key=lambda item: self._candidate_sort_key(
                item,
                paper_section=paper_section,
                code_focus=code_focus,
            ),
            reverse=True,
        )[0]
        evidence = candidate.code_evidence
        plan_steps = current_plan.steps or tuple(trace.step for trace in step_traces)
        return AlignmentResult(
            paper_section_title=paper_section.title,
            code_file_name=evidence.file_name,
            alignment_score=0.55,
            match_type="partial_match",
            analysis="",
            improvement_suggestion="建议继续沿当前逻辑块向上追踪定义与配置来源。",
            retrieval_score=candidate.retrieval_score,
            semantic_evidence="",
            research_supplement="",
            evidence_level=self._derive_evidence_level(evidence, False),
            operator_alignment=self._build_operator_alignment(paper_section, evidence),
            shape_alignment=self._build_shape_alignment(paper_section, evidence),
            highlighted_line_numbers=(
                (evidence.start_line,) if evidence.start_line <= evidence.end_line else ()
            ),
            code_snippet=evidence.code_snippet,
            code_language=evidence.language,
            code_start_line=evidence.start_line,
            code_end_line=evidence.end_line,
            retrieval_plan="\n".join(step.display_text for step in plan_steps),
            implementation_chain=self._build_grounded_implementation_chain(
                paper_section,
                evidence,
                "",
                current_candidates,
            ),
            reflection="这一轮没有拿到稳定的模型总结，我保留了最接近的代码证据。",
            confidence_note="模型本轮响应不稳定，我先给你一个本地兜底结论。",
            agent_observations=tuple(trace.observation for trace in step_traces),
            needs_manual_review=True,
            plan_steps=plan_steps,
            step_traces=step_traces,
        )

    def _has_source_grounding(self, result: AlignmentResult, evidence: CodeEvidence) -> bool:
        return bool(result.implementation_chain.strip()) and not evidence.code_snippet.startswith(
            "# 当前未定位到对应源码"
        )

    def _derive_evidence_level(self, evidence: CodeEvidence, has_definition: bool) -> str:
        if has_definition:
            return "强关联：我已经沿定义链追到源头实现。"
        if evidence.docstring or evidence.symbol_name:
            return "中关联：文件职责和逻辑块语义与论文片段接近。"
        return "弱关联：目前只拿到了局部代码片段，仍需继续核对。"

    def _build_operator_alignment(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> str:
        paper_text = paper_section.combined_text
        code_text = evidence.code_snippet.lower()
        if "softmax" in paper_text and "softmax" in code_text:
            return "论文里的 softmax 在代码中找到了直接对应的 softmax 算子。"
        if "attention" in paper_text and any(
            marker in code_text for marker in ("q_proj", "k_proj", "v_proj", "attn")
        ):
            return "论文中的注意力机制在代码里表现为 q/k/v 投影与注意力权重计算。"
        if "loss" in paper_text and "loss" in code_text:
            return "论文描述的损失项在这段代码里有直接的损失计算痕迹。"
        return "当前代码候选与论文机制存在语义承接，但算子级映射仍需要人工复核。"

    def _build_shape_alignment(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> str:
        paper_text = paper_section.combined_text.lower()
        code_text = evidence.code_snippet.lower()
        if any(marker in code_text for marker in ("reshape", "view(", "permute", "transpose")):
            return (
                "代码里出现了显式的张量重排或维度转换，这通常对应论文中的多头拆分或特征融合步骤。"
            )
        if "head" in paper_text and any(marker in code_text for marker in ("head", "heads")):
            return "论文提到的 head 维度在代码里也有对应的 head 拆分线索。"
        return "当前候选里没有看到足够明确的张量形状处理证据。"

    def _candidate_id(self, candidate: AlignmentCandidate) -> str:
        evidence = candidate.code_evidence
        return f"{evidence.file_name}:{evidence.start_line}-{evidence.end_line}"

    def _candidate_sort_key(
        self,
        candidate: AlignmentCandidate,
        *,
        paper_section: PaperSection,
        code_focus: tuple[str, ...],
    ) -> tuple[float, float, float, float]:
        return (
            self._compute_focus_alignment_score(
                candidate.code_evidence,
                paper_section=paper_section,
                code_focus=code_focus,
            ),
            self._compute_mechanism_alignment_score(
                candidate.code_evidence,
                paper_section=paper_section,
            ),
            self._compute_specificity_score(
                candidate.code_evidence,
                paper_section=paper_section,
            ),
            candidate.retrieval_score,
        )

    def _candidate_priority_score(
        self,
        candidate: AlignmentCandidate,
        *,
        paper_section: PaperSection,
        code_focus: tuple[str, ...],
    ) -> float:
        return (
            self._compute_focus_alignment_score(
                candidate.code_evidence,
                paper_section=paper_section,
                code_focus=code_focus,
            )
            + self._compute_mechanism_alignment_score(
                candidate.code_evidence,
                paper_section=paper_section,
            )
            + self._compute_specificity_score(
                candidate.code_evidence,
                paper_section=paper_section,
            )
            + candidate.retrieval_score
        )

    def _derive_focus_terms(self, paper_section: PaperSection) -> tuple[str, ...]:
        paper_text = f"{paper_section.title} {paper_section.combined_text}".lower()
        focus_terms: list[str] = []
        phrases = (
            "dual-scale",
            "topological map",
            "global action planning",
            "global planning",
            "coarse-scale",
            "fine-scale",
            "cross-modal",
            "graph-aware",
            "position encoding",
            "visited nodes",
            "navigable nodes",
            "current node",
            "pair-wise distance matrix",
            "graph structure",
        )
        for phrase in phrases:
            if phrase in paper_text and phrase not in focus_terms:
                focus_terms.append(phrase)
        single_terms = (
            "navigable",
            "global",
            "local",
            "attention",
            "encoder",
            "position",
            "distance",
            "heading",
        )
        for token in single_terms:
            if token in paper_text and token not in focus_terms:
                focus_terms.append(token)
        if self._is_overview_section(paper_section):
            for token in ("model overview", "navigation", "grounding", "dual-scale"):
                if token not in focus_terms:
                    focus_terms.append(token)
        return tuple(focus_terms)

    def _stabilize_candidates(
        self,
        initial_candidates: tuple[AlignmentCandidate, ...],
        latest_candidates: tuple[AlignmentCandidate, ...],
        *,
        paper_section: PaperSection,
        code_focus: tuple[str, ...],
    ) -> tuple[AlignmentCandidate, ...]:
        merged: dict[str, AlignmentCandidate] = {}
        for candidate in (*initial_candidates, *latest_candidates):
            candidate_id = self._candidate_id(candidate)
            existing = merged.get(candidate_id)
            if existing is None or candidate.retrieval_score > existing.retrieval_score:
                merged[candidate_id] = candidate
        return tuple(
            sorted(
                merged.values(),
                key=lambda item: self._candidate_sort_key(
                    item,
                    paper_section=paper_section,
                    code_focus=code_focus,
                ),
                reverse=True,
            )[:10]
        )

    def _compute_mechanism_alignment_score(
        self,
        evidence: CodeEvidence,
        *,
        paper_section: PaperSection,
    ) -> float:
        paper_text = paper_section.combined_text.lower()
        symbol = (evidence.symbol_name or evidence.parent_symbol or evidence.block_type).lower()
        searchable = " ".join(
            part
            for part in (
                evidence.file_name,
                evidence.symbol_name,
                evidence.parent_symbol,
                evidence.docstring,
                evidence.code_snippet[:600],
            )
            if part
        ).lower()
        score = 0.0

        if self._is_overview_section(paper_section):
            if evidence.block_type == "class":
                score += 2.8
            if "vilmodel.py" in evidence.file_name.lower():
                score += 2.4
            if any(
                term in searchable
                for term in (
                    "glocaltextpathnavcmt",
                    "globalmapencoder",
                    "graphlxrtxlayer",
                    "graphmap",
                    "forward_navigation_per_step",
                )
            ):
                score += 6.6
            if any(
                term in symbol
                for term in (
                    "update_graph",
                    "get_pos_fts",
                    "save_to_json",
                    ".visited",
                    ".distance",
                    ".path",
                )
            ):
                score -= 6.4
            if "graph_utils.py" in evidence.file_name.lower() and evidence.block_type != "class":
                score -= 2.4

        if any(
            term in paper_text
            for term in (
                "topological map",
                "visited nodes",
                "navigable nodes",
                "current node",
                "update et",
                "neighboring unvisited nodes",
            )
        ):
            if "graphmap.update_graph" in symbol or " update_graph(" in searchable:
                score += 8.4
            elif any(
                term in searchable
                for term in ("add_edge", "self.graph.update(", "node_positions", "candidate")
            ):
                score += 5.2
            if symbol == "graphmap":
                score += 5.0
            if symbol.endswith(".__init__"):
                score += 3.8
            if any(term in symbol for term in ("get_pos_fts", "calculate_vp_rel_pos_fts")):
                score += 2.6
            if symbol == "floydgraph":
                score += 1.8
            if "save_to_json" in symbol:
                score -= 4.2
            if symbol.endswith(".path") and not any(
                term in paper_text for term in ("path", "route", "shortest")
            ):
                score -= 3.4
            if symbol.endswith(".visited") and "visited node" in paper_text:
                score -= 3.0
            if any(
                term in evidence.file_name.lower()
                for term in ("/agent", "\\agent", "/reverie/", "\\reverie\\")
            ):
                score -= 2.8

        return score

    def _compute_focus_alignment_score(
        self,
        evidence: CodeEvidence,
        *,
        paper_section: PaperSection,
        code_focus: tuple[str, ...],
    ) -> float:
        searchable = " ".join(
            part
            for part in (
                evidence.file_name,
                evidence.symbol_name,
                evidence.parent_symbol,
                evidence.docstring,
                evidence.code_snippet[:600],
            )
            if part
        ).lower()
        paper_text = paper_section.combined_text.lower()
        score = 0.0
        for focus in code_focus:
            normalized_focus = focus.lower()
            if normalized_focus in searchable:
                score += 3.0
            if normalized_focus in evidence.file_name.lower():
                score += 2.5

        if any(term in paper_text for term in ("global action planning", "global planning")):
            if any(
                term in searchable for term in ("global", "gmap", "graph_sprels", "global_encoder")
            ):
                score += 5.0
        if self._is_overview_section(paper_section):
            if any(
                term in searchable
                for term in (
                    "glocaltextpathnavcmt",
                    "globalmapencoder",
                    "graphlxrtxlayer",
                    "graphmap",
                )
            ):
                score += 6.0
            if "vilmodel.py" in evidence.file_name.lower():
                score += 2.0
            if any(
                term in searchable
                for term in ("update_graph", "get_pos_fts", "save_to_json", ".visited", ".path")
            ):
                score -= 6.0
            if "graph_utils.py" in evidence.file_name.lower() and evidence.block_type != "class":
                score -= 2.0
        if any(term in paper_text for term in ("graph", "graph-aware", "topological map", "map")):
            if any(
                term in searchable
                for term in (
                    "graph",
                    "gmap",
                    "graphmap",
                    "floydgraph",
                    "graph_sprels",
                    "topological",
                )
            ):
                score += 5.0
        if self._is_topological_mapping_section(paper_section):
            if any(
                term in searchable
                for term in (
                    "update_graph",
                    "node_positions",
                    "add_edge",
                    "candidate",
                    "get_pos_fts",
                )
            ):
                score += 6.0
            if any(
                term in searchable
                for term in ("teacher_action", "make_equiv_action", "rollout", "imitation_learning")
            ):
                score -= 8.0
        if evidence.block_type == "module_intro":
            score -= 6.0
        if "coarse-scale" in paper_text and any(
            term in searchable for term in ("global", "graph", "coarse", "gmap")
        ):
            score += 3.0
        if "fine-scale" in paper_text and any(
            term in searchable for term in ("local", "vp", "fine", "panorama")
        ):
            score += 2.0
        if "attention" in paper_text and "attention" in searchable:
            score += 1.5
        if any(term in paper_text for term in ("encoder", "cross-modal", "graph-aware")) and (
            "/models/" in evidence.file_name.lower()
            or evidence.file_name.lower().startswith("models/")
        ):
            score += 3.0

        if any(term in paper_text for term in ("graph", "map", "global planning")) and any(
            term in searchable for term in ("bertselfattention", "transformer class", "adamw")
        ):
            score -= 2.0
        if any(
            term in paper_text
            for term in (
                "topological map",
                "topological mapping",
                "visited nodes",
                "navigable nodes",
            )
        ) and any(
            term in evidence.file_name.lower()
            for term in (
                "/agent",
                "\\agent",
                "/reverie/",
                "\\reverie\\",
                "/parser",
                "\\parser\\",
                "/env.py",
                "\\env.py",
            )
        ):
            score -= 2.6
        if any(term in paper_text for term in ("encoder", "cross-modal", "coarse-scale")) and any(
            term in searchable
            for term in (
                "rollout",
                "make_equiv_action",
                "/agent.py",
                "\\agent.py",
                "/env.py",
                "\\env.py",
                "/agent_",
                "\\agent_",
                "agent_obj",
            )
        ):
            score -= 8.0
        return score

    def _compute_specificity_score(
        self,
        evidence: CodeEvidence,
        *,
        paper_section: PaperSection,
    ) -> float:
        file_name = evidence.file_name.lower()
        paper_text = paper_section.combined_text.lower()
        span = max(1, evidence.end_line - evidence.start_line + 1)
        score = 0.0

        if self._is_trivial_helper_evidence(evidence):
            score -= 3.4
        elif span <= 12:
            score += 0.2
        elif span <= 48:
            score += 1.6
        elif span <= 120:
            score += 1.0
        elif span <= 200:
            score += 0.4
        elif span >= 260:
            score -= 1.6

        if self._is_overview_section(paper_section):
            if evidence.block_type == "class":
                score += 2.4
            if 24 <= span <= 180:
                score += 1.8
            if span <= 12:
                score -= 1.4

        if any(term in paper_text for term in ("encoder", "cross-modal", "coarse-scale")):
            if "/models/" in file_name or file_name.startswith("models/"):
                score += 1.8
            if any(term in file_name for term in ("/agent", "\\agent", "/reverie/", "/r2r/")):
                score -= 2.4

        if any(term in paper_text for term in ("graph", "map", "global planning")) and any(
            term in file_name for term in ("graph_utils", "vilmodel", "transformer")
        ):
            score += 1.0
        if self._is_topological_mapping_section(paper_section):
            if any(term in file_name for term in ("graph_utils", "vilmodel")):
                score += 2.6
            if any(
                term in file_name
                for term in (
                    "/r2r/agent.py",
                    "\\r2r\\agent.py",
                    "/reverie/agent",
                    "\\reverie\\agent",
                )
            ):
                score -= 4.0
            if any(
                term in (evidence.symbol_name or "").lower()
                for term in ("teacher_action", "rollout", "make_equiv_action")
            ):
                score -= 6.0

        return score

    def _is_topological_mapping_section(self, paper_section: PaperSection) -> bool:
        text = paper_section.combined_text.lower()
        return any(
            term in text
            for term in (
                "topological mapping",
                "topological map",
                "graph updating",
                "visited nodes",
                "navigable nodes",
            )
        )

    def _is_overview_section(self, paper_section: PaperSection) -> bool:
        title = paper_section.title.lower()
        text = paper_section.combined_text.lower()
        if any(term in title for term in ("abstract", "overview", "summary")):
            return True
        abstract_markers = (
            "in this work",
            "we propose",
            "our model",
            "significantly outperforms",
            "state-of-the-art",
            "benchmark",
        )
        mechanism_markers = (
            "dual-scale",
            "graph transformer",
            "topological map",
            "cross-modal",
            "global action planning",
        )
        return (
            sum(marker in text for marker in abstract_markers) >= 2
            and sum(marker in text for marker in mechanism_markers) >= 2
        )

    def _is_trivial_helper_evidence(self, evidence: CodeEvidence) -> bool:
        if evidence.block_type not in {"method", "function"}:
            return False

        span = max(1, evidence.end_line - evidence.start_line + 1)
        symbol = (evidence.symbol_name or "").lower()
        snippet_lines = [
            line.strip()
            for line in evidence.code_snippet.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        body_lines = [line for line in snippet_lines if not line.startswith(("def ", "async def "))]
        lowered_body = " ".join(body_lines).lower()

        if span <= 3:
            return True

        if len(body_lines) <= 2 and lowered_body.startswith("return "):
            return True

        trivial_prefixes = ("is_", "has_", "get_", "visited", "distance", "path")
        if span <= 5 and any(part in symbol for part in trivial_prefixes):
            if "update_graph" not in symbol and "get_pos_fts" not in symbol:
                return True

        if span <= 6 and all(
            token not in lowered_body
            for token in (
                "for ",
                "while ",
                "append(",
                "add_edge",
                "graph_sprels",
                "concatenate",
                "stack(",
            )
        ):
            if lowered_body.count("return ") <= 1:
                return True

        return False

    def _build_grounded_implementation_chain(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
        raw_chain: str,
        current_candidates: tuple[AlignmentCandidate, ...],
    ) -> str:
        symbol = evidence.symbol_name or evidence.block_type
        parts = [
            (
                f"当前展示的源码片段来自 `{evidence.file_name}` 的 "
                f"L{evidence.start_line}-L{evidence.end_line}，核心逻辑块是 `{symbol}`。"
            ),
            self._build_module_match_reason(paper_section, evidence),
        ]

        cleaned_chain = raw_chain.strip()
        if self._is_chain_consistent_with_evidence(cleaned_chain, evidence, current_candidates):
            parts.append(cleaned_chain)
        else:
            parts.append(self._build_local_chain_reason(paper_section, evidence))

        return " ".join(part for part in parts if part).strip()

    def _build_module_match_reason(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> str:
        paper_text = paper_section.combined_text
        lowered_paper_text = paper_text.lower()
        searchable = " ".join(
            part
            for part in (
                evidence.file_name,
                evidence.symbol_name,
                evidence.docstring,
                evidence.code_snippet[:400],
            )
            if part
        ).lower()
        explicit_reason = self._build_explicit_relation_reason(
            paper_text,
            lowered_paper_text,
            evidence,
            searchable,
        )
        return explicit_reason

    def _build_explicit_relation_reason(
        self,
        paper_text: str,
        lowered_paper_text: str,
        evidence: CodeEvidence,
        searchable: str,
    ) -> str:
        symbol = evidence.symbol_name or evidence.parent_symbol or evidence.block_type
        symbol_lower = symbol.lower()
        relation_rules = (
            (
                ("get_pos_fts",),
                (
                    "position encoding",
                    "relative to the current node",
                    "heading and distance",
                    "location",
                ),
                "负责把候选节点相对当前位置的方位、距离和步数编码成位置特征",
                "所以它对应的就是论文里“节点如何带上位置编码”这一步。",
            ),
            (
                ("graphlxrtxlayer.forward", "graphlxrtxlayer"),
                (
                    "graph-aware",
                    "pair-wise distance matrix",
                    "graph structure",
                    "global action space",
                ),
                "负责把图结构偏置并入跨模态注意力计算",
                "所以它对应的是论文里图感知自注意力和粗尺度跨模态编码那部分机制。",
            ),
            (
                ("crossmodalencoder", "graphlxrtxlayer"),
                ("multi-layer", "graph-aware cross-modal transformer", "cross-attention layer"),
                "负责把多层跨模态编码层串起来并反复更新节点与文本表示",
                "所以它对应的是论文里多层图感知跨模态 Transformer 的整体编码骨架。",
            ),
            (
                ("save_to_json",),
                ("visited", "navigable", "map", "node"),
                "把地图里的节点、得分和图连接状态整理成可导出的结构",
                "虽然论文不会直接写导出函数，但它把论文里的地图状态变成了可检查的数据表示。",
            ),
            (
                ("update_graph",),
                ("topological map", "visited", "navigable", "builds its own map", "graph updating"),
                "负责把新的观测写回地图、更新节点位置并补齐节点之间的边",
                "所以它对应的是论文里“在线更新拓扑图”的实现部分。",
            ),
            (
                ("floydgraph.path",),
                ("topological map", "global action space", "visited", "navigable"),
                "负责按图里的中间节点递归还原路径",
                "它不是论文里的主机制本身，但支撑了全局规划里依赖的图路径计算。",
            ),
        )

        for symbol_markers, paper_markers, code_role, relation_tail in relation_rules:
            if not any(marker in symbol_lower for marker in symbol_markers):
                continue
            quote = self._extract_paper_quote(paper_text, paper_markers)
            if quote:
                return f"论文中“{quote}”提到了这一步，而 `{symbol}` {code_role}，{relation_tail}"
            if any(marker in lowered_paper_text for marker in paper_markers):
                return f"`{symbol}` {code_role}，{relation_tail}"
        return ""

    def _extract_paper_quote(self, paper_text: str, keywords: tuple[str, ...]) -> str:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。；;])\s+|\n+", paper_text)
            if sentence.strip()
        ]
        if not sentences:
            return ""

        best_sentence = ""
        best_score = float("-inf")
        for sentence in sentences:
            score = self._score_paper_quote_sentence(sentence, keywords)
            if score > best_score:
                best_sentence = sentence
                best_score = score

        if not best_sentence or best_score <= 0:
            return ""
        return self._shorten_quote(best_sentence)

    def _score_paper_quote_sentence(self, sentence: str, keywords: tuple[str, ...]) -> float:
        normalized = " ".join(sentence.split())
        lowered = normalized.lower()
        keyword_hits = sum(1 for keyword in keywords if keyword in lowered)
        if keyword_hits == 0:
            return -1.0

        score = float(keyword_hits * 6)
        word_count = len(normalized.split())
        char_count = len(normalized)
        if self._looks_like_heading_sentence(normalized):
            score -= 9.0
        if word_count < 6:
            score -= 5.0
        elif 8 <= word_count <= 40:
            score += 2.5
        if 40 <= char_count <= 220:
            score += 1.5
        if any(
            token in lowered
            for token in (" we ", " model ", " node", " map", " graph", " attention")
        ):
            score += 1.0
        return score

    def _looks_like_heading_sentence(self, sentence: str) -> bool:
        normalized = sentence.strip()
        if not normalized:
            return True
        if re.fullmatch(r"[\d.]+\s*[A-Za-z][A-Za-z0-9\- ]*", normalized):
            return True
        if re.fullmatch(r"(section|sec\.?)\s*[\d.]+.*", normalized, flags=re.IGNORECASE):
            return True
        words = normalized.split()
        if len(words) <= 4 and all(word[:1].isupper() or word.isdigit() for word in words):
            return True
        return False

    def _shorten_quote(self, sentence: str, limit: int = 160) -> str:
        normalized = " ".join(sentence.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip(" ,，。;；") + "…"

    def _build_local_chain_reason(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> str:
        if evidence.docstring.strip():
            return f"从这段逻辑块的说明看，它主要负责：{evidence.docstring.strip()}"
        operator_alignment = self._build_operator_alignment(paper_section, evidence)
        shape_alignment = self._build_shape_alignment(paper_section, evidence)
        return (
            " ".join(
                item
                for item in (operator_alignment, shape_alignment)
                if item and "仍需要人工复核" not in item
            )
            or "当前先用这段代码作为最接近的实现入口，建议继续沿调用链往上追定义来源。"
        )

    def _is_chain_consistent_with_evidence(
        self,
        chain: str,
        evidence: CodeEvidence,
        current_candidates: tuple[AlignmentCandidate, ...],
    ) -> bool:
        if not chain:
            return False
        lowered_chain = chain.lower()
        current_file = evidence.file_name.lower()
        for candidate in current_candidates:
            other_file = candidate.code_evidence.file_name.lower()
            if other_file != current_file and other_file in lowered_chain:
                return False

        if len(current_candidates) <= 1:
            return True

        evidence_tokens = {
            current_file,
            evidence.symbol_name.lower() if evidence.symbol_name else "",
            evidence.parent_symbol.lower() if evidence.parent_symbol else "",
        }
        evidence_tokens = {token for token in evidence_tokens if len(token) >= 3}
        return any(token in lowered_chain for token in evidence_tokens)

    def _emit(self, handler: AgentEventHandler | None, **payload: object) -> None:
        if handler is not None:
            handler(payload)
