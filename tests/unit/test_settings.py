"""配置模块测试。"""

from pathlib import Path

from labflow.config.settings import Settings


def test_settings_reads_environment_variables(monkeypatch) -> None:
    """环境变量应覆盖默认配置。"""

    env_file = Path(".tmp") / "tests" / "settings-read.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("", encoding="utf-8")

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("API_KEY", "demo-key")
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MODEL_NAME", "demo-model")

    settings = Settings.from_environment(dotenv_path=str(env_file))

    assert settings.app_env == "test"
    assert settings.api_key == "demo-key"
    assert settings.base_url == "https://example.com/v1"
    assert settings.model_name == "demo-model"
    assert settings.has_llm_credentials is True


def test_settings_handles_missing_api_key(monkeypatch) -> None:
    """缺少 API Key 时应识别为未就绪。"""

    env_file = Path(".tmp") / "tests" / "settings-missing-key.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("", encoding="utf-8")

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MODEL_NAME", "demo-model")

    settings = Settings.from_environment(dotenv_path=str(env_file))

    assert settings.has_llm_credentials is False
