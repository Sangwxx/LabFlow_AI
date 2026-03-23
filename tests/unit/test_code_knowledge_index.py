"""代码知识索引测试。"""

from __future__ import annotations

import json

from labflow.reasoning.agent_tools import AgentToolContext, ReasoningToolbox
from labflow.reasoning.code_knowledge_index import CodeKnowledgeIndex
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import CodeEvidence, PaperSection


class _NullLLMClient:
    def generate_json(self, **_: object) -> dict | None:
        return None


class _RerankLLMClient:
    def generate_json(self, **kwargs: object) -> dict | None:
        user_prompt = str(kwargs.get("user_prompt", ""))
        items_start = user_prompt.find("[")
        items_end = user_prompt.find("]\n\n请输出 JSON")
        if items_start < 0 or items_end < 0:
            return None
        items = json.loads(user_prompt[items_start : items_end + 1])
        ranked_items = []
        for item in items:
            symbol = str(item.get("symbol", ""))
            score = 0.3
            if "GraphLXRTXLayer.forward" in symbol:
                score = 0.98
            elif "BertSelfAttention" in symbol:
                score = 0.24
            ranked_items.append(
                {
                    "identity": item["identity"],
                    "score": score,
                    "reason": symbol,
                }
            )
        ranked_items.sort(key=lambda item: item["score"], reverse=True)
        return {"ranked_items": ranked_items}


def test_code_knowledge_index_prefers_mechanism_update_over_helper() -> None:
    """拓扑图章节应优先检索图更新主逻辑，而不是两行 helper。"""

    section = PaperSection(
        title="3.1 Topological Mapping",
        content=(
            "The model gradually builds its own map with visited nodes, navigable nodes "
            "and the current node, then updates graph edges online."
        ),
        level=2,
        page_number=3,
        order=7,
    )
    code_evidences = (
        CodeEvidence(
            file_name="map_nav_src/models/graph_utils.py",
            code_snippet=("def visited(self, k):\n    return k in self._visited\n"),
            related_git_diff="",
            symbols=("visited",),
            commit_context=(),
            start_line=73,
            end_line=74,
            symbol_name="FloydGraph.visited",
            parent_symbol="FloydGraph",
            block_type="method",
        ),
        CodeEvidence(
            file_name="map_nav_src/models/graph_utils.py",
            code_snippet=(
                "def update_graph(self, current_node, navigable_nodes, visited_nodes):\n"
                "    self.node_positions[current_node['viewpoint']] = current_node['position']\n"
                "    for node in navigable_nodes:\n"
                "        self.graph.add_edge(current_node['viewpoint'], node['viewpointId'], node['distance'])\n"
                "    self.graph.update(current_node['viewpoint'])\n"
            ),
            related_git_diff="",
            symbols=("update_graph", "navigable_nodes", "visited_nodes", "add_edge"),
            commit_context=(),
            start_line=106,
            end_line=118,
            symbol_name="GraphMap.update_graph",
            parent_symbol="GraphMap",
            block_type="method",
        ),
    )
    builder = EvidenceBuilder()
    semantic_index = builder.build_semantic_index_from_evidences(code_evidences)

    candidates = CodeKnowledgeIndex(
        semantic_index,
        llm_client=_NullLLMClient(),
    ).search(
        section,
        focus_terms=("topological map", "visited nodes", "navigable nodes", "current node"),
        top_k=2,
        use_llm_rerank=False,
    )

    assert candidates
    assert candidates[0].code_evidence.symbol_name == "GraphMap.update_graph"


def test_code_knowledge_index_can_use_llm_rerank_for_encoder_candidates() -> None:
    """知识索引应允许 LLM 在前排候选里把编码器主逻辑提到前面。"""

    section = PaperSection(
        title="3.2.2 Coarse-scale Cross-modal Encoder",
        content=(
            "Each Transformer layer contains a cross-attention layer and a graph-aware "
            "self-attention layer for global action planning."
        ),
        level=2,
        page_number=4,
        order=8,
    )
    code_evidences = (
        CodeEvidence(
            file_name="pretrain_src/model/vilmodel.py",
            code_snippet=(
                "class BertSelfAttention(nn.Module):\n"
                "    def forward(self, hidden_states, attention_mask):\n"
                "        return hidden_states\n"
            ),
            related_git_diff="",
            symbols=("BertSelfAttention",),
            commit_context=(),
            start_line=79,
            end_line=120,
            symbol_name="BertSelfAttention",
            block_type="class",
        ),
        CodeEvidence(
            file_name="map_nav_src/models/vilmodel.py",
            code_snippet=(
                "def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):\n"
                "    visn_att_output = self.visual_attention(img_embeds, txt_embeds, ctx_att_mask=txt_masks)[0]\n"
                "    if graph_sprels is not None:\n"
                "        img_masks = img_masks + graph_sprels\n"
                "    visn_att_output = self.visn_self_att(visn_att_output, img_masks)[0]\n"
            ),
            related_git_diff="",
            symbols=("GraphLXRTXLayer.forward", "graph_sprels", "visual_attention"),
            commit_context=(),
            start_line=384,
            end_line=395,
            symbol_name="GraphLXRTXLayer.forward",
            parent_symbol="GraphLXRTXLayer",
            block_type="method",
        ),
    )
    builder = EvidenceBuilder()
    semantic_index = builder.build_semantic_index_from_evidences(code_evidences)

    candidates = CodeKnowledgeIndex(
        semantic_index,
        llm_client=_RerankLLMClient(),
    ).search(
        section,
        focus_terms=("coarse-scale", "cross-modal", "graph-aware", "global action planning"),
        top_k=2,
        use_llm_rerank=True,
    )

    assert candidates
    assert candidates[0].code_evidence.symbol_name == "GraphLXRTXLayer.forward"


def test_reasoning_toolbox_semantic_search_uses_knowledge_index() -> None:
    """运行时语义搜索应走知识索引，而不是只做旧的摘要 BM25。"""

    section = PaperSection(
        title="3.2.2 Coarse-scale Cross-modal Encoder",
        content="The graph-aware cross-modal encoder predicts actions in the global action space.",
        level=2,
        page_number=4,
        order=8,
    )
    encoder_evidence = CodeEvidence(
        file_name="map_nav_src/models/vilmodel.py",
        code_snippet=(
            "def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):\n"
            "    visn_att_output = self.visual_attention(img_embeds, txt_embeds, ctx_att_mask=txt_masks)[0]\n"
            "    if graph_sprels is not None:\n"
            "        img_masks = img_masks + graph_sprels\n"
            "    return self.visn_self_att(visn_att_output, img_masks)[0]\n"
        ),
        related_git_diff="",
        symbols=("GraphLXRTXLayer.forward", "graph_sprels", "visual_attention"),
        commit_context=(),
        start_line=384,
        end_line=395,
        symbol_name="GraphLXRTXLayer.forward",
        parent_symbol="GraphLXRTXLayer",
        block_type="method",
    )
    helper_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=("def visited(self, k):\n    return k in self._visited\n"),
        related_git_diff="",
        symbols=("visited",),
        commit_context=(),
        start_line=73,
        end_line=74,
        symbol_name="FloydGraph.visited",
        parent_symbol="FloydGraph",
        block_type="method",
    )
    toolbox = ReasoningToolbox(EvidenceBuilder(), _NullLLMClient())
    context = AgentToolContext(
        paper_section=section,
        project_structure="map_nav_src/models/vilmodel.py",
        code_evidences=(encoder_evidence, helper_evidence),
        current_candidates=(),
    )

    result = toolbox._handle_llm_semantic_search(
        {"query": "graph-aware cross-modal encoder for global action space"},
        context,
    )

    assert result.candidates
    assert result.candidates[0].code_evidence.symbol_name == "GraphLXRTXLayer.forward"
