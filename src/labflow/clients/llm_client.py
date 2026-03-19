"""统一的 LLM 客户端封装。"""

from __future__ import annotations

import json
from importlib import import_module

from labflow.config.settings import get_settings


class LLMClient:
    """我把模型调用细节收口在这里，避免推理层直接碰底层 SDK。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        if not self._settings.has_llm_credentials:
            raise RuntimeError("模型配置不完整，先补齐 `.env` 后再执行对齐分析。")
        self._client = self._build_client()

    def _build_client(self):
        """按需初始化 OpenAI 兼容客户端。"""

        try:
            openai_module = import_module("openai")
        except ModuleNotFoundError as exc:
            raise RuntimeError("当前环境缺少 openai 依赖，先安装后再执行推理。") from exc

        return openai_module.OpenAI(
            api_key=self._settings.api_key,
            base_url=self._settings.base_url,
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> dict:
        """请求模型并解析结构化 JSON 结果。"""

        response = self._client.chat.completions.create(
            model=self._settings.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return self._loads_json(content)

    def _loads_json(self, content: str) -> dict:
        """把模型返回内容解析成 JSON 对象。"""

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            normalized = content.strip()
            normalized = normalized.removeprefix("```json").removeprefix("```").removesuffix("```")
            return json.loads(normalized.strip())
