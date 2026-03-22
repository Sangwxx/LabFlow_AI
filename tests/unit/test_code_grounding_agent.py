"""源码对齐子 Agent 测试。"""

from labflow.reasoning.code_grounding_agent import CodeGroundingAgent
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import AlignmentCandidate, CodeEvidence, PaperSection


class _DummyLLMClient:
    def generate_json(self, **_: object) -> dict | None:
        return None


class _DummyEngine:
    pass


def test_code_grounding_prioritizes_graph_modules_for_global_planning_section() -> None:
    """图规划章节应优先命中 graph / map 相关实现，而不是泛化 attention 壳层。"""

    section = PaperSection(
        title="3.2 Global Action Planning",
        content=(
            "The coarse-scale encoder performs graph-aware planning over a topological map "
            "and predicts actions in the global action space."
        ),
        level=2,
        page_number=4,
        order=8,
    )
    builder = EvidenceBuilder()
    code_evidences = (
        CodeEvidence(
            file_name="pretrain_src/model/vilmodel.py",
            code_snippet=(
                "class BertSelfAttention(nn.Module):\n"
                "    def forward(self, hidden_states, attention_mask):\n"
                "        attention_scores = torch.matmul(query, key.transpose(-1, -2))\n"
            ),
            related_git_diff="",
            symbols=("BertSelfAttention", "attention_scores"),
            commit_context=(),
            start_line=79,
            end_line=120,
            symbol_name="BertSelfAttention",
            block_type="class",
        ),
        CodeEvidence(
            file_name="map_nav_src/models/vilmodel.py",
            code_snippet=(
                "class GraphLXRTXLayer(nn.Module):\n"
                "    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):\n"
                "        if graph_sprels is not None:\n"
                "            visn_attention_mask = visn_attention_mask + graph_sprels\n"
            ),
            related_git_diff="",
            symbols=("GraphLXRTXLayer", "graph_sprels", "visn_attention_mask"),
            commit_context=(),
            start_line=365,
            end_line=392,
            symbol_name="GraphLXRTXLayer",
            block_type="class",
        ),
        CodeEvidence(
            file_name="map_nav_src/models/graph_utils.py",
            code_snippet=(
                "class GraphMap(object):\n"
                "    def update_graph(self, ob):\n"
                "        self.graph.add_edge(ob['viewpoint'], ob['viewpoint'], 0)\n"
            ),
            related_git_diff="",
            symbols=("GraphMap", "update_graph", "graph"),
            commit_context=(),
            start_line=95,
            end_line=112,
            symbol_name="GraphMap",
            block_type="class",
        ),
    )
    semantic_index = builder.build_semantic_index_from_evidences(code_evidences)
    agent = CodeGroundingAgent(_DummyLLMClient(), builder, _DummyEngine())

    candidates = agent._build_initial_candidates(
        paper_section=section,
        code_evidences=code_evidences,
        code_focus=("global planning", "graph", "map", "coarse-scale"),
        semantic_index=semantic_index,
    )

    assert candidates
    assert candidates[0].code_evidence.file_name in {
        "map_nav_src/models/vilmodel.py",
        "map_nav_src/models/graph_utils.py",
    }


def test_grounded_implementation_chain_stays_on_selected_evidence() -> None:
    """源码说明必须锚定当前展示片段，而不是引用其他候选文件。"""

    section = PaperSection(
        title="3.2 Global Action Planning",
        content="The model uses a graph-aware encoder for global planning.",
        level=2,
        page_number=4,
        order=8,
    )
    selected_evidence = CodeEvidence(
        file_name="map_nav_src/models/vilmodel.py",
        code_snippet="class GraphLXRTXLayer(nn.Module):\n    pass\n",
        related_git_diff="",
        symbols=("GraphLXRTXLayer",),
        commit_context=(),
        start_line=365,
        end_line=392,
        symbol_name="GraphLXRTXLayer",
        block_type="class",
    )
    other_evidence = CodeEvidence(
        file_name="pretrain_src/model/vilmodel.py",
        code_snippet="class BertSelfAttention(nn.Module):\n    pass\n",
        related_git_diff="",
        symbols=("BertSelfAttention",),
        commit_context=(),
        start_line=79,
        end_line=141,
        symbol_name="BertSelfAttention",
        block_type="class",
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())

    chain = agent._build_grounded_implementation_chain(
        section,
        selected_evidence,
        "这段实现主要在 pretrain_src/model/vilmodel.py 里完成。",
        (
            AlignmentCandidate(section, selected_evidence, 0.61),
            AlignmentCandidate(section, other_evidence, 0.58),
        ),
    )

    assert "map_nav_src/models/vilmodel.py" in chain
    assert "pretrain_src/model/vilmodel.py" not in chain
