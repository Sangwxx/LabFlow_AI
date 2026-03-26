"""独立论文导读页相关测试。"""

from labflow.ui.guide_page import (
    build_guide_page_overview_html,
    build_guide_page_status_text,
    build_quick_guide_page_header_html,
)
from labflow.ui.paper_preview import LandingPaperPreview


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


def test_build_guide_page_overview_html_hides_raw_abstract() -> None:
    """导读页论文概览不应把原始英文摘要直接当正文展示。"""

    html = build_guide_page_overview_html(build_preview())

    assert "Think Global, Act Local" in html
    assert "Author A" in html
    assert "A dual-scale graph transformer is introduced for navigation." not in html


def test_build_guide_page_status_text_reflects_runtime_state() -> None:
    """导读页顶部应明确说明当前能否直接进入工作区，以及是否启用模型。"""

    assert (
        build_guide_page_status_text(has_repo_path=True, has_llm_credentials=True)
        == "代码目录已连接，可直接进入工作区继续定位源码实现。"
    )
    assert (
        build_guide_page_status_text(has_repo_path=True, has_llm_credentials=False)
        == "代码目录已连接；当前未配置模型 API Key，导读会优先使用本地兜底。"
    )
