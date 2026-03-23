"""阅读代理测试。"""

from labflow.reasoning.learning_agents import ReadingAgent
from labflow.reasoning.models import PaperSection


class FakeTranslationAgent:
    """避免在阅读代理测试里重复触发译文调用。"""

    def reuse_or_translate(self, paper_section: PaperSection, text: str) -> str:
        _ = paper_section
        return text or "这是稳定译文。"


class FakeLearningPacketLLMClient:
    """返回合并后的核心要点与术语百科。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_json(self, *, system_prompt: str, **_: object) -> dict:
        self.calls.append(system_prompt)
        return {
            "core_points": [
                "模型先建立拓扑地图，再在全局节点上做动作预测。",
                "节点表示同时编码位置、视觉观测和历史访问状态。",
                "全局规划模块依赖图距离矩阵来约束注意力计算。",
            ],
            "glossary": [
                "**Topological Map**：按节点和边组织环境结构的全局地图。",
                "**Graph-aware Attention**：把图距离或图偏置并入注意力权重。",
            ],
        }


def test_reading_agent_can_generate_core_points_and_glossary_in_one_call() -> None:
    """阅读代理应优先使用合并请求，减少一次额外的 LLM 往返。"""

    llm_client = FakeLearningPacketLLMClient()
    agent = ReadingAgent(
        llm_client,
        translation_agent=FakeTranslationAgent(),
    )
    section = PaperSection(
        title="3.2 Global Action Planning",
        content="The coarse-scale encoder predicts actions over visited and navigable nodes.",
        level=2,
        page_number=4,
        order=12,
    )

    outputs = agent.run(section, translation="这是稳定译文。")

    assert outputs.translation == "这是稳定译文。"
    assert outputs.core_points.count("- ") == 3
    assert outputs.glossary.count("- ") == 2
    assert len(llm_client.calls) == 1
