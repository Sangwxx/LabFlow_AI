"""统一的 LLM 客户端封装。"""

from __future__ import annotations

import json
import re
import time
from importlib import import_module

from labflow.config.settings import get_settings

REACT_AGENT_ROLE_PROMPT = (
    "你是一个精通 PyTorch 和 Vision-Language Navigation 的源码审计专家。"
    "你不是一个编译器，你是一个导师。"
    "你的任务是根据用户选中的论文片段，利用 list_files 和 read_code 工具在代码库中寻找实现证据。"
    "你的回答必须侧重理论与实践的结合。"
    "即便没有代码支持，你也要告诉用户这段话在讲什么，它的核心创新点在哪里。"
    "你必须通过 Thought 表达推理，通过 Action 调用工具，最后给出 Final Answer。"
)


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
    ) -> dict | None:
        """请求模型并解析结构化 JSON，失败时回退到 `None`。"""

        for attempt in range(3):
            try:
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
            except Exception as exc:  # noqa: BLE001
                if self._is_rate_limit_error(exc) and attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                return None
        return None

    def get_react_agent_role_prompt(self) -> str:
        """给 Agent 家族共享同一份角色设定。"""

        return REACT_AGENT_ROLE_PROMPT

    def _loads_json(self, content: str) -> dict | None:
        """把模型返回内容尽可能清洗成 JSON 对象。"""

        for candidate in self._build_json_candidates(content):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _build_json_candidates(self, content: str) -> tuple[str, ...]:
        """分层清洗模型输出，尽量从脏文本里抠出一个 JSON 对象。"""

        normalized = content.strip()
        normalized = (
            normalized.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        without_control_chars = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", normalized)
        object_slice = self._extract_json_object(without_control_chars)

        candidates = [content, normalized, without_control_chars]
        if object_slice:
            candidates.append(object_slice)
        return tuple(dict.fromkeys(item for item in candidates if item))

    def _extract_json_object(self, content: str) -> str:
        """尝试从混杂文本中裁出最外层 JSON 对象。"""

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return content[start : end + 1]

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """兼容不同 SDK 版本的限流异常识别。"""

        class_name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        return "ratelimit" in class_name or "rate limit" in message or "429" in message
