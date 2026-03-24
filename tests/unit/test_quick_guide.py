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
    assert "未见环境下的泛化" in guide.problem
    assert "双尺度图 Transformer" in guide.method
    assert "完整链路" in guide.abstract_digest


def test_build_landing_quick_guide_prefers_llm_result() -> None:
    """模型返回结构化导读时，应优先展示中文结果。"""

    class FakeLLMClient:
        def generate_json(self, **_: object) -> dict:
            return {
                "headline": "先确认任务设定，再看它如何把建图和决策拆成两层。",
                "problem": "论文主要在解决陌生环境里的视觉语言导航问题。",
                "method": "方法核心是把拓扑建图和全局动作规划放进双尺度图结构里。",
                "focus": "第一次阅读时优先看摘要、方法图和 3.1/3.2 两节。",
                "abstract_digest": (
                    "摘要主要在说明任务背景、双尺度图结构以及它如何同时覆盖建图和动作规划这两部分。"
                ),
            }

    guide = build_landing_quick_guide(build_preview(), llm_client=FakeLLMClient())

    assert guide == LandingQuickGuide(
        headline="先确认任务设定，再看它如何把建图和决策拆成两层",
        problem="论文主要在解决陌生环境里的视觉语言导航问题",
        method="方法核心是把拓扑建图和全局动作规划放进双尺度图结构里",
        focus="第一次阅读时优先看摘要、方法图和 3.1/3.2 两节",
        abstract_digest="摘要主要在说明任务背景、双尺度图结构以及它如何同时覆盖建图和动作规划这两部分",
    )


def test_build_quick_guide_html_contains_sections() -> None:
    """快速导读 HTML 应包含三段结构化说明。"""

    html = build_quick_guide_html(
        LandingQuickGuide(
            headline="一屏看懂论文主线。",
            problem="解决什么问题。",
            method="方法核心是什么。",
            focus="第一次阅读时先看哪里。",
            abstract_digest="中文摘要导读。",
        )
    )

    assert "这篇论文要解决什么" in html
    assert "方法核心" in html
    assert "阅读时先关注什么" in html


def test_coerce_landing_quick_guide_replaces_english_cached_result() -> None:
    """历史缓存里如果还是英文结果，应回退成中文兜底版本。"""

    class LegacyGuide:
        headline = "Read the paper first."
        problem = "The paper studies navigation."
        method = "It uses a graph transformer."
        focus = "Read the method section."

    guide = coerce_landing_quick_guide(LegacyGuide(), build_preview())

    assert "视觉语言导航" in guide.headline
    assert "未见环境下的泛化" in guide.problem
    assert "双尺度图 Transformer" in guide.method
    assert guide.abstract_digest


def test_coerce_landing_quick_guide_replaces_low_information_result() -> None:
    """即便是中文结果，只要内容明显空泛，也应回退到更具体的导读。"""

    guide = coerce_landing_quick_guide(
        LandingQuickGuide(
            headline="先用中文导读抓住问题、方法和阅读重点。",
            problem="这篇论文围绕任务展开。",
            method="从现有信息看，方法核心在于组织信息流。",
            focus="第一次阅读时先看摘要。",
            abstract_digest="当前还没有拿到稳定的中文摘要导读。",
        ),
        build_preview(),
    )

    assert "视觉语言导航" in guide.headline
    assert "双尺度图 Transformer" in guide.method
    assert "完整链路" in guide.abstract_digest
