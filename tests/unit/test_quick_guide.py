"""首页快速导读相关测试。"""

from labflow.ui.paper_preview import LandingPaperPreview
from labflow.ui.quick_guide import (
    LandingQuickGuide,
    build_landing_quick_guide,
    build_quick_guide_html,
    coerce_landing_quick_guide,
)


def build_preview() -> LandingPaperPreview:
    """构造用于首页导读的论文预览数据。"""

    return LandingPaperPreview(
        title="Think Global, Act Local",
        authors=("Author A", "Author B"),
        abstract=(
            "This paper studies vision-and-language navigation in unseen environments. "
            "It proposes a dual-scale graph transformer for topological mapping "
            "and global action planning."
        ),
        source_label="已识别论文 · Think.pdf",
        meta_items=("15 页", "CVPR 2022"),
        external_url="https://example.com/paper",
    )


def test_build_landing_quick_guide_falls_back_to_chinese_summary() -> None:
    """没有模型时，快速导读也应优先返回中文导读内容。"""

    guide = build_landing_quick_guide(build_preview(), llm_client=None)

    assert "视觉语言导航" in guide.headline
    assert "未见环境下的泛化" in guide.core_question
    assert "双尺度图 Transformer" in guide.core_conclusion
    assert "Introduction" in guide.intro_path
    assert "完整链路" in guide.abstract_digest
    assert guide.core_question.count("。") >= 2
    assert guide.takeaway.count("。") >= 2


def test_build_landing_quick_guide_prefers_llm_result() -> None:
    """模型返回结构化导读时，应优先展示中文结果。"""

    class FakeLLMClient:
        def generate_json(self, **_: object) -> dict:
            return {
                "headline": "先确认任务设定，再看它如何把建图和决策拆成两层。",
                "core_question": "论文主要在解决陌生环境里的视觉语言导航问题。",
                "core_conclusion": "作者认为双尺度图结构能同时覆盖建图和动作规划这两部分。",
                "contribution": "关键贡献是把拓扑建图和全局动作规划统一进同一套双尺度框架。",
                "intro_path": "Introduction 先交代任务，再指出旧方法不足，最后引出双尺度结构。",
                "data_method": "方法上重点看双尺度图结构，实验上重点看 REVERIE 和 SOON 这些基准。",
                "analysis_flow": "实验阅读时先看主结果，再看消融和成功率指标是否一起提升。",
                "takeaway": "最值得学习的是先拆问题，再让模块设计逐一对应难点。",
                "limitation": "还需要继续看失败案例和额外实验，判断方法边界是否足够清楚。",
                "abstract_digest": (
                    "摘要主要在说明任务背景、双尺度图结构以及它如何同时覆盖建图和动作规划这两部分。"
                ),
            }

    guide = build_landing_quick_guide(build_preview(), llm_client=FakeLLMClient())

    assert guide == LandingQuickGuide(
        headline="先确认任务设定，再看它如何把建图和决策拆成两层",
        core_question="论文主要在解决陌生环境里的视觉语言导航问题",
        core_conclusion="作者认为双尺度图结构能同时覆盖建图和动作规划这两部分",
        contribution="关键贡献是把拓扑建图和全局动作规划统一进同一套双尺度框架",
        intro_path="Introduction 先交代任务，再指出旧方法不足，最后引出双尺度结构",
        data_method="方法上重点看双尺度图结构，实验上重点看 REVERIE 和 SOON 这些基准",
        analysis_flow="实验阅读时先看主结果，再看消融和成功率指标是否一起提升",
        takeaway="最值得学习的是先拆问题，再让模块设计逐一对应难点",
        limitation="还需要继续看失败案例和额外实验，判断方法边界是否足够清楚",
        abstract_digest="摘要主要在说明任务背景、双尺度图结构以及它如何同时覆盖建图和动作规划这两部分",
    )


def test_clean_quick_guide_text_keeps_longer_report_style_content() -> None:
    """导读报告模式下，不应把正常长度的段落过早截断。"""

    guide = build_landing_quick_guide(build_preview(), llm_client=None)

    assert len(guide.core_question) > 40
    assert len(guide.abstract_digest) > 60


def test_build_quick_guide_html_contains_sections() -> None:
    """快速导读 HTML 应包含三段结构化说明。"""

    html = build_quick_guide_html(
        LandingQuickGuide(
            headline="一屏看懂论文主线。",
            core_question="解决什么问题。",
            core_conclusion="结论是什么。",
            contribution="贡献是什么。",
            intro_path="引言怎么展开。",
            data_method="方法与实验怎么搭。",
            analysis_flow="分析怎么组织。",
            takeaway="能学到什么。",
            limitation="还有什么不足。",
            abstract_digest="摘要导读。",
        )
    )

    assert "核心研究问题" in html
    assert "核心结论" in html
    assert "Introduction 怎么展开" in html
    assert "摘要导读" in html


def test_coerce_landing_quick_guide_replaces_english_cached_result() -> None:
    """历史缓存里如果还是英文结果，应回退成中文兜底版本。"""

    class LegacyGuide:
        headline = "Read the paper first."
        problem = "The paper studies navigation."
        method = "It uses a graph transformer."
        focus = "Read the method section."

    guide = coerce_landing_quick_guide(LegacyGuide(), build_preview())

    assert "视觉语言导航" in guide.headline
    assert "未见环境下的泛化" in guide.core_question
    assert "双尺度图 Transformer" in guide.core_conclusion
    assert guide.abstract_digest


def test_coerce_landing_quick_guide_replaces_low_information_result() -> None:
    """即便是中文结果，只要内容明显空泛，也应回退到更具体的导读。"""

    guide = coerce_landing_quick_guide(
        LandingQuickGuide(
            headline="先用中文导读抓住问题、方法和阅读重点。",
            core_question="这篇论文围绕任务展开。",
            core_conclusion="先看结论。",
            contribution="有一些贡献。",
            intro_path="引言会介绍背景。",
            data_method="从现有信息看，方法核心在于组织信息流。",
            analysis_flow="先看实验。",
            takeaway="有帮助。",
            limitation="还可以继续研究。",
            abstract_digest="当前还没有拿到稳定的中文摘要导读。",
        ),
        build_preview(),
    )

    assert "视觉语言导航" in guide.headline
    assert "双尺度图 Transformer" in guide.core_conclusion
    assert "完整链路" in guide.abstract_digest


def test_coerce_landing_quick_guide_supports_legacy_cached_instance() -> None:
    """旧缓存里同名对象缺少新字段时，也应自动转成当前版本结构。"""

    legacy_guide = object.__new__(LandingQuickGuide)
    object.__setattr__(legacy_guide, "headline", "Read the paper first.")
    object.__setattr__(legacy_guide, "problem", "The paper studies navigation.")
    object.__setattr__(legacy_guide, "method", "It uses a graph transformer.")
    object.__setattr__(legacy_guide, "focus", "Read the method section.")
    object.__setattr__(legacy_guide, "abstract_digest", "Old cache summary.")

    guide = coerce_landing_quick_guide(legacy_guide, build_preview())

    assert "视觉语言导航" in guide.headline
    assert guide.core_question
    assert guide.core_conclusion
