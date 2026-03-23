"""源码导览摘要生成测试。"""

from labflow.reasoning.code_grounding_agent import CodeGroundingAgent
from labflow.reasoning.evidence_builder import EvidenceBuilder
from labflow.reasoning.models import CodeEvidence, PaperSection


class _DummyLLMClient:
    def generate_json(self, **_: object) -> dict | None:
        return None


class _DummyEngine:
    pass


def _make_agent() -> CodeGroundingAgent:
    return CodeGroundingAgent(_DummyLLMClient(), EvidenceBuilder(), _DummyEngine())


def test_guide_summary_prefers_concrete_code_behavior_explanation() -> None:
    """源码导览摘要应优先解释代码具体做了什么。"""

    evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):\n"
            "    rel_angles, rel_dists = [], []\n"
            "    rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)\n"
            "    return np.concatenate([rel_ang_fts, rel_dists], 1)\n"
        ),
        related_git_diff="",
        symbols=("get_pos_fts", "get_angle_fts"),
        commit_context=(),
        start_line=127,
        end_line=148,
        symbol_name="GraphMap.get_pos_fts",
        parent_symbol="GraphMap",
        block_type="method",
    )

    summary = _make_agent()._build_guide_summary(
        evidence,
        semantic_summary="定义了 get_pos_fts，调用了 get_angle_fts。",
    )

    assert "计算相对朝向" in summary
    assert "位置特征向量" in summary


def test_guide_summary_explains_init_as_state_setup() -> None:
    evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def __init__(self, start_vp):\n"
            "    self.start_vp = start_vp\n"
            "    self.node_positions = {}\n"
            "    self.graph = FloydGraph()\n"
            "    self.node_embeds = {}\n"
            "    self.node_nav_scores = {}\n"
        ),
        related_git_diff="",
        symbols=("FloydGraph",),
        commit_context=(),
        start_line=96,
        end_line=104,
        symbol_name="GraphMap.__init__",
        parent_symbol="GraphMap",
        block_type="method",
    )

    summary = _make_agent()._build_guide_summary(evidence)

    assert "初始化" in summary
    assert "`GraphMap`" in summary
    assert "`node_positions`" in summary
    assert "更新图边" in summary


def test_guide_summary_explains_serialization_behavior() -> None:
    evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def save_to_json(self):\n"
            "    nodes = []\n"
            "    for vp, pos in self.node_positions.items():\n"
            '        nodes.append({"vp": vp, "pos": pos, "visited": vp in self.graph.visited})\n'
            '    return {"nodes": nodes, "edges": self.graph._dis}\n'
        ),
        related_git_diff="",
        symbols=("node_positions", "graph"),
        commit_context=(),
        start_line=150,
        end_line=168,
        symbol_name="GraphMap.save_to_json",
        parent_symbol="GraphMap",
        block_type="method",
    )

    summary = _make_agent()._build_guide_summary(evidence)

    assert "可保存的结构" in summary
    assert "导出" in summary
    assert "复现" in summary


def test_guide_summary_explains_recursive_path_reconstruction() -> None:
    evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def path(self, x, y):\n"
            '    if self._dis[x][y] == float("inf"):\n'
            "        return []\n"
            '    if self._point[x][y] == "":\n'
            "        return [y]\n"
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

    summary = _make_agent()._build_guide_summary(evidence)

    assert "递归" in summary
    assert "路径" in summary
    assert "拼接" in summary


def test_guide_summary_explains_global_map_input_embedding() -> None:
    evidence = CodeEvidence(
        file_name="map_nav_src/models/vilmodel.py",
        code_snippet=(
            "def gmap_input_embedding(self, split_traj_embeds, split_traj_vp_lens, traj_vpids, "
            "traj_cand_vpids, gmap_vpids, gmap_step_ids, gmap_pos_fts, gmap_lens):\n"
            "    gmap_img_fts = self._aggregate_gmap_features(\n"
            "        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids\n"
            "    )\n"
            "    gmap_embeds = gmap_img_fts + self.gmap_step_embeddings(gmap_step_ids) + "
            "self.gmap_pos_embeddings(gmap_pos_fts)\n"
            "    gmap_masks = gen_seq_masks(gmap_lens)\n"
            "    return gmap_embeds, gmap_masks\n"
        ),
        related_git_diff="",
        symbols=("gmap_input_embedding",),
        commit_context=(),
        start_line=612,
        end_line=623,
        symbol_name="GlobalMapEncoder.gmap_input_embedding",
        parent_symbol="GlobalMapEncoder",
        block_type="method",
    )

    summary = _make_agent()._build_guide_summary(evidence)

    assert "全局地图节点" in summary
    assert "导航步编码" in summary
    assert "位置编码" in summary


def test_guide_summary_explains_navigation_fusion_step() -> None:
    evidence = CodeEvidence(
        file_name="map_nav_src/models/vilmodel.py",
        code_snippet=(
            "def forward_navigation_per_step(self, txt_embeds, txt_masks, gmap_img_embeds, "
            "gmap_step_ids, gmap_pos_fts, gmap_masks, gmap_pair_dists, gmap_visited_masks, "
            "gmap_vpids, vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids):\n"
            "    gmap_embeds = gmap_img_embeds + self.global_encoder.gmap_step_embeddings(gmap_step_ids) + "
            "self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)\n"
            "    vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)\n"
            "    global_logits = self.global_sap_head(gmap_embeds).squeeze(2)\n"
            "    local_logits = self.local_sap_head(vp_embeds).squeeze(2)\n"
            "    fused_logits = torch.clone(global_logits)\n"
            "    obj_logits = self.og_head(vp_embeds).squeeze(2)\n"
            "    outs = {'fused_logits': fused_logits, 'obj_logits': obj_logits}\n"
            "    return outs\n"
        ),
        related_git_diff="",
        symbols=("forward_navigation_per_step",),
        commit_context=(),
        start_line=750,
        end_line=831,
        symbol_name="GlocalTextPathNavCMT.forward_navigation_per_step",
        parent_symbol="GlocalTextPathNavCMT",
        block_type="method",
    )

    summary = _make_agent()._build_guide_summary(evidence)

    assert "单步导航决策" in summary
    assert "全局地图分支" in summary
    assert "局部视角分支" in summary
    assert "fused" not in summary.lower()


def test_guide_relevance_reason_quotes_paper_and_connects_to_code_behavior() -> None:
    section = PaperSection(
        title="3.2.2 Coarse-scale Cross-modal Encoder",
        content=(
            "Position encoding embeds node location in the map from an egocentric point of view, "
            "namely relative heading and distance to the current node."
        ),
        level=2,
        page_number=4,
        order=8,
    )
    evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):\n"
            "    rel_angles, rel_dists = [], []\n"
            "    return np.concatenate([rel_ang_fts, rel_dists], 1)\n"
        ),
        related_git_diff="",
        symbols=("get_pos_fts",),
        commit_context=(),
        start_line=127,
        end_line=148,
        symbol_name="GraphMap.get_pos_fts",
        parent_symbol="GraphMap",
        block_type="method",
    )

    reason = _make_agent()._build_guide_relevance_reason(section, evidence)

    assert "论文中“" in reason
    assert "Position encoding embeds node location" in reason
    assert "位置特征" in reason
    assert "位置编码" in reason


def test_guide_relevance_reason_for_serialization_explains_supporting_role() -> None:
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
    evidence = CodeEvidence(
        file_name="map_nav_src/models/graph_utils.py",
        code_snippet=(
            "def save_to_json(self):\n"
            "    for vp, pos in self.node_positions.items():\n"
            '        nodes.append({"vp": vp, "pos": pos, "visited": vp in self.graph.visited})\n'
            '    return {"nodes": nodes, "edges": self.graph._dis}\n'
        ),
        related_git_diff="",
        symbols=("save_to_json",),
        commit_context=(),
        start_line=150,
        end_line=168,
        symbol_name="GraphMap.save_to_json",
        parent_symbol="GraphMap",
        block_type="method",
    )

    reason = _make_agent()._build_guide_relevance_reason(section, evidence)

    assert "论文中“" in reason
    assert "visited nodes and navigable nodes" in reason
    assert "可检查的数据表示" in reason


def test_extract_paper_quote_prefers_body_sentence_over_section_heading() -> None:
    paper_text = (
        "3.1 Topological Mapping\n"
        "The environment graph G is initially unknown to the agent.\n"
        "The model gradually builds its own map using visited nodes and navigable nodes "
        "observed from the current location."
    )

    quote = _make_agent()._extract_paper_quote(
        paper_text,
        ("topological map", "visited", "navigable", "builds its own map"),
    )

    assert quote == (
        "The model gradually builds its own map using visited nodes and navigable nodes "
        "observed from the current location."
    )
