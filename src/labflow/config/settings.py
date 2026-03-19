"""项目配置加载。"""

from dataclasses import dataclass
from functools import lru_cache
from os import environ
from pathlib import Path


def load_dotenv(dotenv_path: str = ".env") -> None:
    """从本地 `.env` 文件加载环境变量。"""

    env_file = Path(dotenv_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    """统一管理运行期配置。"""

    app_name: str = "LabFlow AI"
    app_env: str = "dev"
    api_key: str | None = None
    base_url: str = "https://api.qnaigc.com/v1"
    model_name: str = "moonshotai/kimi-k2.5"

    @property
    def has_llm_credentials(self) -> bool:
        """判断当前是否已经配置模型访问凭据。"""

        return bool(self.api_key and self.base_url and self.model_name)

    @classmethod
    def from_environment(cls, dotenv_path: str = ".env") -> "Settings":
        """根据当前环境变量构建配置对象。"""

        load_dotenv(dotenv_path)
        return cls(
            app_env=environ.get("APP_ENV", "dev"),
            api_key=environ.get("API_KEY"),
            base_url=environ.get("BASE_URL", "https://api.qnaigc.com/v1"),
            model_name=environ.get("MODEL_NAME", "moonshotai/kimi-k2.5"),
        )


@lru_cache
def get_settings() -> Settings:
    """获取缓存后的项目配置。"""

    return Settings.from_environment()
