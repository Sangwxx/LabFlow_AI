"""独立论文导读页相关测试。"""

from labflow.ui.guide_page import (
    build_quick_guide_page_header_html,
    build_quick_guide_reading_steps,
)
from labflow.ui.paper_preview import LandingPaperPreview
from labflow.ui.quick_guide import LandingQuickGuide


def build_preview() -> LandingPaperPreview:
    """构造导读页测试用论文预览。"""

    return LandingPaperPreview(
        title="Think Global, Act Local",
        authors=("Author A", "Author B"),
        abstract="A dual-scale graph transformer is introduced for navigation.",
        source_label="已识别论文 · Think.pdf",
        meta_items=("15 页", "CVPR 2022"),
        external_url=None,
    )


def test_build_quick_guide_page_header_html_contains_title_and_source() -> None:
    """导读页头部应明确告诉用户当前正在看哪篇论文。"""

    html = build_quick_guide_page_header_html(source_name="Think.pdf")

    assert "论文导读" in html
    assert "Think.pdf" in html
    assert "LabFlow" in html


def test_build_quick_guide_reading_steps_prefers_generated_guide() -> None:
    """如果已有导读结论，阅读建议应优先复用导读页生成内容。"""

    steps = build_quick_guide_reading_steps(
        preview=build_preview(),
        guide=LandingQuickGuide(
            headline="一屏看懂论文主线。",
            problem="先理解它要解决的导航问题。",
            method="再看双尺度图结构如何拆分建图和决策。",
            focus="最后重点读方法图和 3.1/3.2 两节。",
        ),
    )

    assert steps == (
        "先确认：先理解它要解决的导航问题。",
        "接着理解：再看双尺度图结构如何拆分建图和决策。",
        "最后重点追踪：最后重点读方法图和 3.1/3.2 两节。",
    )


def test_build_quick_guide_reading_steps_has_fallback_steps() -> None:
    """没有导读结论时，导读页也应保留稳定的阅读顺序建议。"""

    steps = build_quick_guide_reading_steps(preview=build_preview(), guide=None)

    assert len(steps) == 3
    assert "Think Global, Act Local" in steps[0]
    assert "进入工作区" in steps[2]
