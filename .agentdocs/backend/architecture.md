# 后端架构与约束

## 适用范围

适用于 `src/labflow/reasoning/`、`src/labflow/parsers/`、`src/labflow/clients/` 下的推理、解析与模型调用代码。

## 当前架构

- `parsers/` 负责 PDF 与仓库解析，产出结构化输入。
- `reasoning/evidence_builder.py` 负责把仓库切成可检索证据、项目结构与语义索引。
- `reasoning/agent_executor.py` 负责总控编排。
- `reasoning/agent_engine.py` 负责 Planner / Executor / RePlanner 的运行循环。
- `reasoning/learning_agents.py` 负责翻译与导读子 Agent。
- `reasoning/code_grounding_agent.py` 负责源码对齐、候选收束与最终解释。
- `clients/llm_client.py` 统一承接大模型请求。

## 开发约束

- 新功能优先复用现有候选池、语义摘要和计划执行结果，不重复建立平行检索链路。
- 面向 UI 输出的新字段应收敛在 `reasoning/models.py`，避免把页面专属结构散落在多个模块。
- 推理结果允许带兜底信息，但兜底结果也必须结构化、可展示、可测试。
- 需要补充启发式排序时，应优先放在 `CodeGroundingAgent` 的候选收束逻辑里，而不是在 UI 层做二次筛选。
- 新增字段若会长期复用，必须在模型层保留明确中文注释和默认值。

## 测试要求

- 涉及候选排序、结果组装、字段扩展时，补充 `tests/unit/test_code_grounding_agent.py` 或 `tests/unit/test_agent_runtime.py`。
- 保持 `pytest` 可直接覆盖主要分支，不依赖在线接口才能验证核心逻辑。
