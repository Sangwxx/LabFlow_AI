"""源码对齐子 Agent 测试。"""

from labflow.reasoning.code_grounding_agent import CodeGroundingAgent
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import (
    AlignmentCandidate,
    AlignmentResult,
    CodeEvidence,
    ExecutionPlan,
    PaperSection,
    PlanStep,
    StepExecutionTrace,
)


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


def test_initial_candidates_prefer_mechanism_block_over_trivial_helper_in_topology_section() -> (
    None
):
    """最小 RAG 接入后，初始候选也应更偏向机制实现而不是两行 helper。"""

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
    builder = EvidenceBuilder()
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
    semantic_index = builder.build_semantic_index_from_evidences(code_evidences)
    agent = CodeGroundingAgent(_DummyLLMClient(), builder, _DummyEngine())

    candidates = agent._build_initial_candidates(
        paper_section=section,
        code_evidences=code_evidences,
        code_focus=("topological map", "visited nodes", "navigable nodes", "current node"),
        semantic_index=semantic_index,
    )

    assert candidates
    assert candidates[0].code_evidence.symbol_name == "GraphMap.update_graph"


def test_source_guide_for_topological_mapping_skips_teacher_action_blocks() -> None:
    """源码导览在拓扑建图段落里应优先展示图更新主干，而不是任务层 teacher action。"""

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
    builder = EvidenceBuilder()
    teacher_action = CodeEvidence(
        file_name="map_nav_src/r2r/agent.py",
        code_snippet=(
            "def _teacher_action_r4r(self, obs, vpids, ended):\n"
            "    a = np.zeros(len(obs), dtype=np.int64)\n"
            "    for i, ob in enumerate(obs):\n"
            "        if ended[i]:\n"
            "            continue\n"
            "    return a\n"
        ),
        related_git_diff="",
        symbols=("teacher_action", "enumerate", "zeros"),
        commit_context=(),
        start_line=239,
        end_line=285,
        symbol_name="GMapNavAgent._teacher_action_r4r",
        parent_symbol="GMapNavAgent",
        block_type="method",
    )
    update_graph = CodeEvidence(
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
    )
    pos_fts = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):\n"
            "    rel_angles, rel_dists = [], []\n"
            "    for vp in gmap_vpids:\n"
            "        rel_angles.append([0, 0])\n"
            "    return np.concatenate([rel_ang_fts, rel_dists], 1)\n"
        ),
        related_git_diff="",
        symbols=("get_pos_fts", "rel_angles", "rel_dists"),
        commit_context=(),
        start_line=127,
        end_line=148,
        symbol_name="GraphMap.get_pos_fts",
        parent_symbol="GraphMap",
        block_type="method",
    )
    semantic_index = builder.build_semantic_index_from_evidences(
        (teacher_action, update_graph, pos_fts)
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), builder, _DummyEngine())

    guide_candidates = agent._select_guide_candidates(
        paper_section=section,
        current_candidates=(
            AlignmentCandidate(section, teacher_action, 8.5),
            AlignmentCandidate(section, update_graph, 3.1),
            AlignmentCandidate(section, pos_fts, 2.9),
        ),
        semantic_index=semantic_index,
    )

    assert guide_candidates
    assert guide_candidates[0].code_evidence.symbol_name == "GraphMap.update_graph"
    assert all(
        candidate.code_evidence.symbol_name != "GMapNavAgent._teacher_action_r4r"
        for candidate in guide_candidates[:2]
    )


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


def test_stabilized_candidates_keep_graph_modules_ahead_of_generic_agent_blocks() -> None:
    """执行阶段即使返回高召回的 agent 候选，也不该把图规划主候选完全冲掉。"""

    section = PaperSection(
        title="3.2.2 Coarse-scale Cross-modal Encoder",
        content=(
            "The module takes the coarse-scale map Gt and encoded instruction to make "
            "navigation predictions in the global action space with graph-aware attention."
        ),
        level=2,
        page_number=4,
        order=8,
    )
    graph_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet="class GraphMap(object):\n    pass\n",
        related_git_diff="",
        symbols=("GraphMap",),
        commit_context=(),
        start_line=95,
        end_line=168,
        symbol_name="GraphMap",
        block_type="class",
    )
    agent_evidence = CodeEvidence(
        file_name="map_nav_src/reverie/agent_obj.py",
        code_snippet="class GMapObjectNavAgent(Seq2SeqAgent):\n    pass\n",
        related_git_diff="",
        symbols=("GMapObjectNavAgent",),
        commit_context=(),
        start_line=30,
        end_line=498,
        symbol_name="GMapObjectNavAgent",
        block_type="class",
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())

    stabilized = agent._stabilize_candidates(
        (AlignmentCandidate(section, graph_evidence, 2.2804),),
        (AlignmentCandidate(section, agent_evidence, 14.51),),
        paper_section=section,
        code_focus=("global action planning", "coarse-scale", "graph-aware", "map"),
    )

    assert stabilized
    assert stabilized[0].code_evidence.file_name == "map_nav_src/models/graph_utils.py"


def test_cross_modal_encoder_prefers_encoder_block_over_graph_path_utility() -> None:
    """同样是图相关候选时，跨模态编码器应优先命中真正的编码层而不是路径工具函数。"""

    section = PaperSection(
        title="3.2.2 Coarse-scale Cross-modal Encoder",
        content=(
            "The module takes the coarse-scale map Gt and encoded instruction to make "
            "navigation predictions in the global action space. Each Transformer layer "
            "contains a cross-attention layer and a graph-aware self-attention layer."
        ),
        level=2,
        page_number=4,
        order=8,
    )
    graph_path_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def path(self, x, y):\n"
            '    """Return the path from x to y."""\n'
            "    if x == y:\n"
            "        return []\n"
        ),
        related_git_diff="",
        symbols=("path",),
        commit_context=(),
        start_line=76,
        end_line=92,
        symbol_name="FloydGraph.path",
        parent_symbol="FloydGraph",
        block_type="method",
        docstring="Return the path from x to y.",
    )
    encoder_evidence = CodeEvidence(
        file_name="map_nav_src/models/vilmodel.py",
        code_snippet=(
            "def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,\n"
            "            graph_sprels=None):\n"
            "    visn_att_output = self.visual_attention(visn_feats, lang_feats, ctx_att_mask=lang_attention_mask)[0]\n"
            "    if graph_sprels is not None:\n"
            "        visn_attention_mask = visn_attention_mask + graph_sprels\n"
            "    visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]\n"
        ),
        related_git_diff="",
        symbols=("GraphLXRTXLayer.forward", "graph_sprels", "visual_attention", "visn_self_att"),
        commit_context=(),
        start_line=384,
        end_line=395,
        symbol_name="GraphLXRTXLayer.forward",
        parent_symbol="GraphLXRTXLayer",
        block_type="method",
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())

    stabilized = agent._stabilize_candidates(
        (
            AlignmentCandidate(section, graph_path_evidence, 1.52),
            AlignmentCandidate(section, encoder_evidence, 1.04),
        ),
        (),
        paper_section=section,
        code_focus=(
            "global action planning",
            "coarse-scale",
            "graph-aware",
            "cross-modal",
            "encoder",
        ),
    )

    assert stabilized
    assert stabilized[0].code_evidence.symbol_name == "GraphLXRTXLayer.forward"


def test_enriched_result_contains_project_context_and_multiple_source_guides() -> None:
    """源码导览应保留项目结构摘要，并给出多个相关实现模块。"""

    section = PaperSection(
        title="3.2.2 Coarse-scale Cross-modal Encoder",
        content=(
            "The module takes the coarse-scale map Gt and encoded instruction to make "
            "navigation predictions in the global action space with graph-aware attention."
        ),
        level=2,
        page_number=4,
        order=8,
    )
    builder = EvidenceBuilder()
    agent = CodeGroundingAgent(_DummyLLMClient(), builder, _DummyEngine())
    encoder_evidence = CodeEvidence(
        file_name="map_nav_src/models/vilmodel.py",
        code_snippet=(
            "def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,\n"
            "            graph_sprels=None):\n"
            "    visn_att_output = self.visual_attention(visn_feats, lang_feats, ctx_att_mask=lang_attention_mask)[0]\n"
            "    if graph_sprels is not None:\n"
            "        visn_attention_mask = visn_attention_mask + graph_sprels\n"
        ),
        related_git_diff="",
        symbols=("GraphLXRTXLayer.forward", "graph_sprels"),
        commit_context=(),
        start_line=384,
        end_line=395,
        symbol_name="GraphLXRTXLayer.forward",
        parent_symbol="GraphLXRTXLayer",
        block_type="method",
    )
    graph_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "class GraphMap(object):\n"
            "    def update_graph(self, ob):\n"
            "        self.graph.add_edge(ob['viewpoint'], ob['viewpoint'], 0)\n"
        ),
        related_git_diff="",
        symbols=("GraphMap", "update_graph"),
        commit_context=(),
        start_line=95,
        end_line=112,
        symbol_name="GraphMap",
        block_type="class",
    )
    candidates = (
        AlignmentCandidate(section, encoder_evidence, 1.84),
        AlignmentCandidate(section, graph_evidence, 1.55),
    )
    semantic_index = builder.build_semantic_index_from_evidences(
        (encoder_evidence, graph_evidence),
    )
    result = AlignmentResult(
        paper_section_title=section.title,
        code_file_name=encoder_evidence.file_name,
        alignment_score=0.82,
        match_type="strong_match",
        analysis="译文",
        improvement_suggestion="",
        retrieval_score=1.84,
        implementation_chain="当前展示的源码片段来自 map_nav_src/models/vilmodel.py。",
        code_snippet=encoder_evidence.code_snippet,
        code_start_line=encoder_evidence.start_line,
        code_end_line=encoder_evidence.end_line,
    )

    enriched = agent._enrich_result(
        result,
        paper_section=section,
        current_candidates=candidates,
        semantic_index=semantic_index,
        project_structure="\n".join(
            (
                "map_nav_src / models / vilmodel.py",
                "map_nav_src / models / graph_utils.py",
                "map_nav_src / reverie / agent_obj.py",
                "pretrain_src / model / vilmodel.py",
            )
        ),
    )

    assert enriched.project_structure_context
    assert "项目主干目录" in enriched.project_structure_context[0]
    assert len(enriched.source_guide) == 2
    assert enriched.source_guide[0].symbol_name == "GraphLXRTXLayer.forward"
    assert enriched.source_guide[1].symbol_name == "GraphMap"


def test_source_guide_filters_out_trivial_helper_candidates() -> None:
    """源码导览不应把只有两三行的 helper 方法当成主阅读入口。"""

    section = PaperSection(
        title="3.1 Topological Mapping",
        content=(
            "The model gradually builds its own map using visited nodes and navigable nodes "
            "observed from the current location."
        ),
        level=2,
        page_number=3,
        order=7,
    )
    builder = EvidenceBuilder()
    agent = CodeGroundingAgent(_DummyLLMClient(), builder, _DummyEngine())
    helper_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=("def visited(self, k):\n    return (k in self._visited)\n"),
        related_git_diff="",
        symbols=("visited",),
        commit_context=(),
        start_line=73,
        end_line=74,
        symbol_name="FloydGraph.visited",
        parent_symbol="FloydGraph",
        block_type="method",
    )
    mechanism_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def update_graph(self, ob):\n"
            "    self.node_positions[ob['viewpoint']] = ob['position']\n"
            "    for cc in ob['candidate']:\n"
            "        self.node_positions[cc['viewpointId']] = cc['position']\n"
            "        dist = calc_position_distance(ob['position'], cc['position'])\n"
            "        self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], dist)\n"
            "    self.graph.update(ob['viewpoint'])\n"
        ),
        related_git_diff="",
        symbols=("update_graph", "add_edge"),
        commit_context=(),
        start_line=106,
        end_line=118,
        symbol_name="GraphMap.update_graph",
        parent_symbol="GraphMap",
        block_type="method",
    )
    candidates = (
        AlignmentCandidate(section, helper_evidence, 3.2),
        AlignmentCandidate(section, mechanism_evidence, 2.1),
    )
    semantic_index = builder.build_semantic_index_from_evidences(
        (helper_evidence, mechanism_evidence)
    )
    result = AlignmentResult(
        paper_section_title=section.title,
        code_file_name=mechanism_evidence.file_name,
        alignment_score=0.77,
        match_type="strong_match",
        analysis="译文",
        improvement_suggestion="",
        retrieval_score=2.1,
        implementation_chain="当前展示的源码片段来自 map_nav_src/models/graph_utils.py。",
        code_snippet=mechanism_evidence.code_snippet,
        code_start_line=mechanism_evidence.start_line,
        code_end_line=mechanism_evidence.end_line,
    )

    enriched = agent._enrich_result(
        result,
        paper_section=section,
        current_candidates=candidates,
        semantic_index=semantic_index,
        project_structure="map_nav_src / models / graph_utils.py",
    )

    assert enriched.source_guide
    assert all(item.symbol_name != "FloydGraph.visited" for item in enriched.source_guide)
    assert enriched.source_guide[0].symbol_name == "GraphMap.update_graph"


def test_specificity_score_penalizes_trivial_helper_blocks() -> None:
    """两三行的 getter / accessor 不应因为块很小就压过机制实现。"""

    section = PaperSection(
        title="3.1 Topological Mapping",
        content="The model gradually builds its own map using visited nodes and navigable nodes.",
        level=2,
        page_number=3,
        order=7,
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())
    helper_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=("def visited(self, k):\n    return (k in self._visited)\n"),
        related_git_diff="",
        symbols=("visited",),
        commit_context=(),
        start_line=73,
        end_line=74,
        symbol_name="FloydGraph.visited",
        parent_symbol="FloydGraph",
        block_type="method",
    )
    mechanism_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def update_graph(self, ob):\n"
            "    self.node_positions[ob['viewpoint']] = ob['position']\n"
            "    for cc in ob['candidate']:\n"
            "        self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], 0)\n"
            "    self.graph.update(ob['viewpoint'])\n"
        ),
        related_git_diff="",
        symbols=("update_graph", "add_edge"),
        commit_context=(),
        start_line=106,
        end_line=114,
        symbol_name="GraphMap.update_graph",
        parent_symbol="GraphMap",
        block_type="method",
    )

    helper_score = agent._compute_specificity_score(helper_evidence, paper_section=section)
    mechanism_score = agent._compute_specificity_score(mechanism_evidence, paper_section=section)

    assert helper_score < mechanism_score


def test_topological_mapping_prefers_graph_update_over_export_and_path_helpers() -> None:
    """拓扑图构建章节应优先展示图更新机制，而不是导出或路径工具函数。"""

    section = PaperSection(
        title="3.1 Topological Mapping",
        content=(
            "The environment graph G is initially unknown to the agent. There are three types "
            "of nodes in Vt: visited nodes, navigable nodes, and the current node. At each "
            "step t, we add the current node and its neighboring unvisited nodes and update Et."
        ),
        level=2,
        page_number=3,
        order=7,
    )
    focus = ("topological map construction", "graph node management", "visited node tracking")
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())
    update_graph = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def update_graph(self, ob):\n"
            "    self.node_positions[ob['viewpoint']] = ob['position']\n"
            "    for cc in ob['candidate']:\n"
            "        self.node_positions[cc['viewpointId']] = cc['position']\n"
            "        self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], 0)\n"
            "    self.graph.update(ob['viewpoint'])\n"
        ),
        related_git_diff="",
        symbols=("update_graph", "add_edge", "node_positions"),
        commit_context=(),
        start_line=106,
        end_line=112,
        symbol_name="GraphMap.update_graph",
        parent_symbol="GraphMap",
        block_type="method",
    )
    save_to_json = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def save_to_json(self):\n"
            "    nodes = {}\n"
            "    for vp, pos in self.node_positions.items():\n"
            "        nodes[vp] = pos\n"
            "    return nodes\n"
        ),
        related_git_diff="",
        symbols=("save_to_json", "node_positions"),
        commit_context=(),
        start_line=150,
        end_line=168,
        symbol_name="GraphMap.save_to_json",
        parent_symbol="GraphMap",
        block_type="method",
    )
    graph_path = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def path(self, x, y):\n"
            "    k = self._point[x][y]\n"
            "    return self.path(x, k) + self.path(k, y)\n"
        ),
        related_git_diff="",
        symbols=("path",),
        commit_context=(),
        start_line=76,
        end_line=92,
        symbol_name="FloydGraph.path",
        parent_symbol="FloydGraph",
        block_type="method",
    )

    ranked = sorted(
        (
            AlignmentCandidate(section, save_to_json, 7.6),
            AlignmentCandidate(section, graph_path, 1.6),
            AlignmentCandidate(section, update_graph, 0.5),
        ),
        key=lambda item: agent._candidate_sort_key(
            item,
            paper_section=section,
            code_focus=focus,
        ),
        reverse=True,
    )

    assert ranked[0].code_evidence.symbol_name == "GraphMap.update_graph"


def test_explicit_relation_reason_uses_symbol_specific_rule() -> None:
    """关系说明不能让 save_to_json 的模板污染到 path 或位置编码函数。"""

    section = PaperSection(
        title="3.1 Topological Mapping",
        content=(
            "There are three types of nodes in Vt: visited nodes, navigable nodes, and the current node. "
            "At each step t, we add the current node and its neighboring unvisited nodes and update Et."
        ),
        level=2,
        page_number=3,
        order=7,
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())
    path_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def path(self, x, y):\n"
            "    k = self._point[x][y]\n"
            "    return self.path(x, k) + self.path(k, y)\n"
        ),
        related_git_diff="",
        symbols=("path",),
        commit_context=(),
        start_line=76,
        end_line=92,
        symbol_name="FloydGraph.path",
        parent_symbol="FloydGraph",
        block_type="method",
    )

    reason = agent._build_guide_relevance_reason(section, path_evidence)

    assert "递归还原路径" in reason
    assert "可检查的数据表示" not in reason


def test_source_guide_promotes_helper_candidate_to_representative_method() -> None:
    """即使当前候选只有 helper，小函数也应提升到同类里的代表性机制实现。"""

    section = PaperSection(
        title="3.1 Topological Mapping",
        content=(
            "The environment graph G is initially unknown to the agent. At each step t, "
            "we add the current node and its neighboring unvisited nodes and update Et."
        ),
        level=2,
        page_number=3,
        order=7,
    )
    helper_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=("def visited(self, k):\n    return (k in self._visited)\n"),
        related_git_diff="",
        symbols=("visited",),
        commit_context=(),
        start_line=73,
        end_line=74,
        symbol_name="FloydGraph.visited",
        parent_symbol="FloydGraph",
        block_type="method",
    )
    path_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def path(self, x, y):\n"
            "    k = self._point[x][y]\n"
            "    return self.path(x, k) + self.path(k, y)\n"
        ),
        related_git_diff="",
        symbols=("path",),
        commit_context=(),
        start_line=76,
        end_line=92,
        symbol_name="FloydGraph.path",
        parent_symbol="FloydGraph",
        block_type="method",
    )
    update_graph = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def update_graph(self, ob):\n"
            "    self.node_positions[ob['viewpoint']] = ob['position']\n"
            "    for cc in ob['candidate']:\n"
            "        self.node_positions[cc['viewpointId']] = cc['position']\n"
            "        self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], 0)\n"
            "    self.graph.update(ob['viewpoint'])\n"
        ),
        related_git_diff="",
        symbols=("update_graph", "add_edge", "node_positions"),
        commit_context=(),
        start_line=106,
        end_line=112,
        symbol_name="GraphMap.update_graph",
        parent_symbol="GraphMap",
        block_type="method",
    )
    graph_map = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "class GraphMap(object):\n"
            "    def __init__(self):\n"
            "        self.node_positions = {}\n"
            "    def update_graph(self, ob):\n"
            "        self.graph.add_edge(ob['viewpoint'], ob['viewpoint'], 0)\n"
        ),
        related_git_diff="",
        symbols=("GraphMap", "update_graph"),
        commit_context=(),
        start_line=95,
        end_line=148,
        symbol_name="GraphMap",
        block_type="class",
    )
    builder = EvidenceBuilder()
    semantic_index = builder.build_semantic_index_from_evidences(
        (helper_evidence, path_evidence, update_graph, graph_map)
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), builder, _DummyEngine())

    selected = agent._select_guide_candidates(
        paper_section=section,
        current_candidates=(AlignmentCandidate(section, helper_evidence, 3.1),),
        semantic_index=semantic_index,
    )

    assert selected
    assert selected[0].code_evidence.symbol_name == "GraphMap.update_graph"


def test_local_fallback_result_uses_derived_focus_terms_and_prefers_mechanism_code() -> None:
    """fallback 结果不能因为缺失 focus 推导而崩溃，也不该选到两行 helper。"""

    section = PaperSection(
        title="3.1 Topological Mapping",
        content=(
            "The environment graph G is initially unknown to the agent. At each step t, "
            "we add the current node and its neighboring unvisited nodes and update Et."
        ),
        level=2,
        page_number=3,
        order=7,
    )
    helper_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=("def visited(self, k):\n    return (k in self._visited)\n"),
        related_git_diff="",
        symbols=("visited",),
        commit_context=(),
        start_line=73,
        end_line=74,
        symbol_name="FloydGraph.visited",
        parent_symbol="FloydGraph",
        block_type="method",
    )
    update_graph = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def update_graph(self, ob):\n"
            "    self.node_positions[ob['viewpoint']] = ob['position']\n"
            "    for cc in ob['candidate']:\n"
            "        self.node_positions[cc['viewpointId']] = cc['position']\n"
            "        self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], 0)\n"
            "    self.graph.update(ob['viewpoint'])\n"
        ),
        related_git_diff="",
        symbols=("update_graph", "add_edge", "node_positions"),
        commit_context=(),
        start_line=106,
        end_line=112,
        symbol_name="GraphMap.update_graph",
        parent_symbol="GraphMap",
        block_type="method",
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())

    result = agent._build_local_fallback_result(
        paper_section=section,
        current_candidates=(
            AlignmentCandidate(section, helper_evidence, 3.5),
            AlignmentCandidate(section, update_graph, 0.5),
        ),
        step_traces=(
            StepExecutionTrace(
                step=PlanStep(step_id="1", description="定位拓扑图更新逻辑"),
                thought="当前优先排除两行 helper。",
                action="保留图更新主逻辑。",
                observation="候选集中有多个图相关实现。",
                tool_invocations=(),
            ),
        ),
        current_plan=ExecutionPlan(
            steps=(PlanStep(step_id="1", description="定位拓扑图更新逻辑"),),
        ),
    )

    assert result.code_file_name == "map_nav_src/models/graph_utils.py"
    assert result.code_start_line == 106
    assert result.code_end_line == 112


def test_overview_section_prefers_model_level_candidate_over_local_helper() -> None:
    """摘要/总览段应更偏向模型主干，而不是局部 helper。"""

    section = PaperSection(
        title="Abstract",
        content=(
            "In this work, we propose a dual-scale graph transformer for vision-and-language "
            "navigation. Our model combines topological maps with fine-grained cross-modal "
            "understanding and significantly outperforms prior methods on benchmark datasets."
        ),
        level=1,
        page_number=1,
        order=1,
    )
    model_evidence = CodeEvidence(
        file_name="map_nav_src/models/vilmodel.py",
        code_snippet=(
            "class GlocalTextPathNavCMT(nn.Module):\n"
            "    def forward_navigation_per_step(self, batch):\n"
            "        return self.global_encoder(batch)\n"
        ),
        related_git_diff="",
        symbols=("GlocalTextPathNavCMT", "forward_navigation_per_step", "global_encoder"),
        commit_context=(),
        start_line=430,
        end_line=520,
        symbol_name="GlocalTextPathNavCMT",
        block_type="class",
    )
    helper_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def update_graph(self, ob):\n"
            "    self.graph.add_edge(ob['viewpoint'], ob['viewpoint'], 0)\n"
        ),
        related_git_diff="",
        symbols=("update_graph", "add_edge"),
        commit_context=(),
        start_line=106,
        end_line=112,
        symbol_name="GraphMap.update_graph",
        parent_symbol="GraphMap",
        block_type="method",
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())

    ranked = sorted(
        (
            AlignmentCandidate(section, helper_evidence, 1.9),
            AlignmentCandidate(section, model_evidence, 1.2),
        ),
        key=lambda item: agent._candidate_sort_key(
            item,
            paper_section=section,
            code_focus=agent._derive_focus_terms(section),
        ),
        reverse=True,
    )

    assert ranked[0].code_evidence.symbol_name == "GlocalTextPathNavCMT"


def test_short_circuit_grounding_for_clear_topology_candidate() -> None:
    """拓扑建图章节如果头部候选足够强，就应该直接短路收束。"""

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
    strong_evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def update_graph(self, ob):\n"
            "    self.node_positions[ob['viewpoint']] = ob['position']\n"
            "    for cc in ob['candidate']:\n"
            "        self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], 0)\n"
            "    self.graph.update(ob['viewpoint'])\n"
        ),
        related_git_diff="",
        symbols=("update_graph", "add_edge", "node_positions"),
        commit_context=(),
        start_line=106,
        end_line=118,
        symbol_name="GraphMap.update_graph",
        parent_symbol="GraphMap",
        block_type="method",
    )
    weak_evidence = CodeEvidence(
        file_name="map_nav_src/reverie/agent_obj.py",
        code_snippet="def rollout(self):\n    return None\n",
        related_git_diff="",
        symbols=("rollout",),
        commit_context=(),
        start_line=30,
        end_line=31,
        symbol_name="GMapObjectNavAgent.rollout",
        parent_symbol="GMapObjectNavAgent",
        block_type="method",
    )
    agent = CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())

    assert (
        agent._should_short_circuit_grounding(
            paper_section=section,
            initial_candidates=(
                AlignmentCandidate(section, strong_evidence, 2.8),
                AlignmentCandidate(section, weak_evidence, 0.2),
            ),
            code_focus=("topological map", "visited nodes", "navigable nodes"),
        )
        is True
    )
