"""首页快速导读相关测试。"""

from labflow.ui.paper_preview import LandingPaperPreview
from labflow.ui.quick_guide import (
    LandingQuickGuide,
    build_landing_quick_guide,
    build_quick_guide_html,
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


def test_build_landing_quick_guide_falls_back_to_local_abstract() -> None:
    """没有模型时，首页快速导读也应能基于摘要给出稳定兜底。"""

    guide = build_landing_quick_guide(build_preview(), llm_client=None)

    assert "先用一屏内容抓住问题" in guide.headline
    assert "vision-and-language navigation" in guide.problem
    assert "dual-scale graph transformer" in guide.method
    assert "第一次阅读时" in guide.focus


def test_build_landing_quick_guide_prefers_llm_result() -> None:
    """模型返回结构化导读时，应优先展示更像人写的结果。"""

    class FakeLLMClient:
        def generate_json(self, **_: object) -> dict:
            return {
                "headline": "先确认任务设定，再看它如何把建图和决策拆成两层。",
                "problem": "论文主要在解决陌生环境里的视觉语言导航问题。",
                "method": "方法核心是把拓扑建图和全局动作规划放进双尺度图结构里。",
                "focus": "第一次阅读时优先看摘要、方法图和 3.1/3.2 两节。",
            }

    guide = build_landing_quick_guide(build_preview(), llm_client=FakeLLMClient())

    assert guide == LandingQuickGuide(
        headline="先确认任务设定，再看它如何把建图和决策拆成两层",
        problem="论文主要在解决陌生环境里的视觉语言导航问题",
        method="方法核心是把拓扑建图和全局动作规划放进双尺度图结构里",
        focus="第一次阅读时优先看摘要、方法图和 3.1/3.2 两节",
    )


def test_build_quick_guide_html_contains_sections() -> None:
    """首页快速导读 HTML 应包含三段结构化说明。"""

    html = build_quick_guide_html(
        LandingQuickGuide(
            headline="一屏看懂论文主线。",
            problem="解决什么问题。",
            method="方法核心是什么。",
            focus="第一次阅读时先看哪里。",
        )
    )

    assert "快速导读" in html
    assert "这篇论文要解决什么" in html
    assert "方法核心" in html
    assert "阅读时先关注什么" in html
