"""LLM 客户端回归测试。"""

from labflow.clients.llm_client import LLMClient


def test_llm_client_loads_json_from_code_fence() -> None:
    """模型把 JSON 包在代码块里时，我也要能稳稳抠出来。"""

    client = LLMClient.__new__(LLMClient)

    payload = client._loads_json('```json\n{"answer": 1}\n```')

    assert payload == {"answer": 1}


def test_llm_client_loads_json_after_control_char_cleanup() -> None:
    """脏控制字符不应该再把整个页面带崩。"""

    client = LLMClient.__new__(LLMClient)

    payload = client._loads_json('{"answer":"ok\x01"}')

    assert payload == {"answer": "ok"}


def test_llm_client_returns_none_for_invalid_json() -> None:
    """彻底无效的 JSON 应回退到 None，让上层统一降级。"""

    client = LLMClient.__new__(LLMClient)

    payload = client._loads_json("not-json-at-all")

    assert payload is None
