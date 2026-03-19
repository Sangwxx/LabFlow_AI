"""首页内容构建。"""

from dataclasses import dataclass

from labflow.config.settings import Settings


@dataclass(frozen=True)
class StageCard:
    """首页阶段卡片。"""

    step: str
    title: str
    description: str


@dataclass(frozen=True)
class HomeContent:
    """首页展示内容。"""

    title: str
    subtitle: str
    stage_cards: tuple[StageCard, ...]
    current_scope: tuple[str, ...]
    next_actions: tuple[str, ...]


def build_home_content(settings: Settings) -> HomeContent:
    """构建首页展示所需的文案与结构。"""

    readiness = (
        "模型配置已就绪，可进入解析与对齐实现。"
        if settings.has_llm_credentials
        else ("模型配置未完成，当前可先推进本地解析与 BM25 检索链路。")
    )

    return HomeContent(
        title="LabFlow AI",
        subtitle=(
            "面向实验室场景的论文-代码对齐分析助手，"
            "首版聚焦文本型 PDF 与本地 Git 仓库的可追溯关联。"
        ),
        stage_cards=(
            StageCard("Perceive", "感知输入", "解析论文 PDF、读取本地仓库结构并建立统一中间表示。"),
            StageCard(
                "Retrieve",
                "检索证据",
                "优先采用 BM25 兜底检索，为后续本地 embedding 留出扩展位。",
            ),
            StageCard("Reason", "对齐推理", "围绕章节、方法、模块与证据片段生成可解释的对齐判断。"),
            StageCard("Act", "生成结论", "输出 Markdown 审计报告，为第二阶段周报能力打基础。"),
        ),
        current_scope=(
            "首版只支持文本型 PDF，不处理扫描版 OCR。",
            "Git 仓库输入采用本地路径导入，优先保证稳定性。",
            readiness,
        ),
        next_actions=(
            "实现 PDF 章节提取与结构清洗。",
            "实现本地 Git 仓库树、提交记录与关键文件摘要。",
            "打通对齐结果的证据链展示与 Markdown 报告导出。",
        ),
    )
