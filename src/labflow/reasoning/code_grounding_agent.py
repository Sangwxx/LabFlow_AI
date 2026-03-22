from __future__ import annotations

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
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import (
    AlignmentCandidate,
    AlignmentResult,
    CodeEvidence,
    CodeSemanticSummary,
    ExecutionPlan,
    PaperSection,
    StepExecutionTrace,
)


class CodeGroundingAgent:
    """我只负责论文片段与源码逻辑的对齐和解释。"""

    EXECUTION_BUDGET_SEC = 28.0

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
        semantic_index = self._evidence_builder.build_semantic_index_from_evidences(code_evidences)
        self._emit_semantic_cards(semantic_index, event_handler)
        current_candidates = self._build_initial_candidates(
            paper_section=paper_section,
            code_evidences=code_evidences,
            code_focus=plan.code_focus,
            semantic_index=semantic_index,
        )
        if not current_candidates:
            self._emit(
                event_handler,
                kind="observation",
                message="当前没有定位到足够可信的源码候选，本轮先只保留论文导读结果。",
            )
            return None

        current_plan, step_traces, current_candidates = self._engine.run(
            paper_section=paper_section,
            project_structure=project_structure,
            code_evidences=code_evidences,
            current_candidates=current_candidates,
            role_prompt=role_prompt,
            event_handler=event_handler,
            max_runtime_sec=self.EXECUTION_BUDGET_SEC,
        )
        if (
            all(not trace.tool_invocations for trace in step_traces)
            or current_plan.final_summary == "执行阶段达到时限，我先用已有候选收束结果。"
        ):
            self._emit(
                event_handler,
                kind="observation",
                message="执行阶段没有稳定收敛到足够强的工具证据，我直接基于当前候选生成兜底结论。",
            )
            result = self._build_local_fallback_result(
                paper_section,
                current_candidates,
                step_traces,
                current_plan,
            )
            reflected_result = result
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
        return reflected_result

    def _build_initial_candidates(
        self,
        *,
        paper_section: PaperSection,
        code_evidences: tuple[CodeEvidence, ...],
        code_focus: tuple[str, ...],
        semantic_index: tuple[CodeSemanticSummary, ...],
    ) -> tuple[AlignmentCandidate, ...]:
        merged: dict[str, AlignmentCandidate] = {}

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
        ) -> None:
            lexical_candidates = self._evidence_builder.build_alignment_candidates_from_inputs(
                paper_sections=(section,),
                code_evidences=code_evidences,
                top_k=6,
            )
            semantic_candidates = self._evidence_builder.retrieve_semantic_candidates(
                section,
                semantic_index,
                top_k=6,
            )
            collect(lexical_candidates, lexical_boost)
            collect(semantic_candidates, semantic_boost)

        collect_for_section(paper_section, lexical_boost=0.02, semantic_boost=0.12)
        for focus in code_focus[:3]:
            synthetic_section = PaperSection(
                title=paper_section.title,
                content=focus,
                level=paper_section.level,
                page_number=paper_section.page_number,
                order=paper_section.order,
                block_orders=paper_section.block_orders,
            )
            collect_for_section(synthetic_section, lexical_boost=0.0, semantic_boost=0.09)

        traced_symbols = self._extract_trace_symbols(code_focus, semantic_index, merged)
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
                    code_focus=code_focus,
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
        paper_text = paper_section.combined_text.lower()
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
        candidate = current_candidates[0]
        evidence = candidate.code_evidence
        plan_steps = current_plan.steps or tuple(trace.step for trace in step_traces)
        symbol = evidence.symbol_name or evidence.block_type
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
        paper_text = paper_section.combined_text.lower()
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
    ) -> tuple[float, float]:
        return (
            self._compute_focus_alignment_score(
                candidate.code_evidence,
                paper_section=paper_section,
                code_focus=code_focus,
            ),
            candidate.retrieval_score,
        )

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
            if any(term in searchable for term in ("global", "gmap", "graph_sprels", "global_encoder")):
                score += 5.0
        if any(term in paper_text for term in ("graph", "graph-aware", "topological map", "map")):
            if any(
                term in searchable
                for term in ("graph", "gmap", "graphmap", "floydgraph", "graph_sprels", "topological")
            ):
                score += 5.0
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
            "/models/" in evidence.file_name.lower() or evidence.file_name.lower().startswith("models/")
        ):
            score += 3.0

        if any(term in paper_text for term in ("graph", "map", "global planning")) and any(
            term in searchable for term in ("bertselfattention", "transformer class", "adamw")
        ):
            score -= 2.0
        if any(term in paper_text for term in ("encoder", "cross-modal", "coarse-scale")) and any(
            term in searchable
            for term in ("rollout", "make_equiv_action", "/agent.py", "\\agent.py", "/env.py", "\\env.py")
        ):
            score -= 8.0
        return score

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
        paper_text = paper_section.combined_text.lower()
        searchable = " ".join(
            part
            for part in (evidence.file_name, evidence.symbol_name, evidence.docstring, evidence.code_snippet[:400])
            if part
        ).lower()
        if any(term in paper_text for term in ("graph", "global planning", "topological map")) and any(
            term in searchable
            for term in ("graph", "gmap", "graphmap", "graph_sprels", "global_encoder", "floydgraph")
        ):
            return "它命中的是图结构或全局规划相关实现，不是单纯的通用注意力壳层。"
        if "attention" in paper_text and any(
            marker in searchable for marker in ("attention", "q_proj", "k_proj", "v_proj")
        ):
            return "它命中的是注意力计算相关实现，和论文里的跨模态编码细节有直接机制重叠。"
        return "这段逻辑块在模块职责和语义关键词上，和当前论文片段最接近。"

    def _build_local_chain_reason(
        self,
        paper_section: PaperSection,
        evidence: CodeEvidence,
    ) -> str:
        if evidence.docstring.strip():
            return f"从这段逻辑块的说明看，它主要负责：{evidence.docstring.strip()}"
        operator_alignment = self._build_operator_alignment(paper_section, evidence)
        shape_alignment = self._build_shape_alignment(paper_section, evidence)
        return " ".join(
            item
            for item in (operator_alignment, shape_alignment)
            if item and "仍需要人工复核" not in item
        ) or "当前先用这段代码作为最接近的实现入口，建议继续沿调用链往上追定义来源。"

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
