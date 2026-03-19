"""首页内容集成测试。"""

from labflow.config.settings import Settings
from labflow.ui.home_content import build_home_content


def test_home_content_matches_current_mvp_scope() -> None:
    """首页文案应体现当前 MVP 决策。"""

    settings = Settings(
        app_env="test",
        api_key="demo-key",
        base_url="https://example.com/v1",
        model_name="demo-model",
    )

    content = build_home_content(settings)

    assert len(content.stage_cards) == 4
    assert content.stage_cards[0].step == "Perceive"
    assert any("本地路径导入" in item for item in content.current_scope)
    assert any("文本型 PDF" in item for item in content.current_scope)
    assert any("Markdown" in item for item in content.next_actions)
