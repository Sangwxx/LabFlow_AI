# 任务文档：重构 Agent 运行时为 Engine / Prompts / Tools 三层

## 任务目标

在不破坏当前论文阅读工作区功能的前提下，把推理层从“单文件塞满全部逻辑”的实现方式，重构为更接近 LangChain 风格的三层架构：

1. `Prompts`：所有 Prompt 模板与消息拼装逻辑独立存放。
2. `Tools`：工具以注册表方式管理，不再在执行器里硬编码分支。
3. `Engine`：用清晰的 `while loop` 统一驱动 Planner / Executor / RePlanner。

## 当前问题

- `src/labflow/reasoning/agent_executor.py` 同时承担：
  - Planner / Executor / RePlanner
  - Tool 调用
  - Prompt 拼装
  - 学术导读模式
  - 结果清洗与兜底
  - 事件流输出
- 当前文件体量过大，已经明显超出单文件应承载的复杂度。
- Prompt 文本分散在主程序里，后续很难独立调试和迭代。
- Tool 调度靠 `if/elif` 分支硬编码，不利于后续继续加工具。
- 运行时虽然已经是 Plan-and-Execute，但还没有形成“Engine / Registry / Prompt 契约”这种可维护结构。

## 重构目标

### 目标架构

```text
reasoning/
├── agent_executor.py      # 兼容入口与高层编排
├── agent_engine.py        # While loop 运行时
├── agent_prompts.py       # Planner / Executor / RePlanner / 学习助手 Prompt
├── agent_tools.py         # Tool 定义、上下文、注册表、默认工具集
├── aligner.py             # 兼容批量 / 单段入口
├── evidence_builder.py    # 代码证据构建
└── models.py              # 数据模型
```

### 兼容要求

- `PlanAndExecuteAgent` 仍保留为外部稳定入口。
- `PaperCodeAligner` 与 `ui/app.py` 不需要感知这次底层拆分。
- 现有测试入口尽量不改或少改。
- 学术导读模式的串行链路必须保留：
  1. 中文译文
  2. 核心要点
  3. 术语百科

## 分阶段计划

### 阶段 1：抽 Prompt

- 把 Planner / Executor / RePlanner / Final Answer / Reflection / 学习助手 Prompt 抽离到独立模块
- 主程序只传结构化上下文，不直接拼长字符串

### 阶段 2：抽 Tool Registry

- 引入 `AgentTool` 与 `ToolRegistry`
- 注册默认工具：
  - `list_project_structure`
  - `read_code_segment`
  - `llm_semantic_search`
  - `find_definition`

### 阶段 3：抽 Engine

- 用独立 `Engine` 承载 `while plan.steps:` 的执行循环
- Engine 只关心：
  - 当前计划
  - 当前步骤
  - 调用工具
  - 回写 Observation
  - 请求 RePlanner

### 阶段 4：兼容迁移

- `PlanAndExecuteAgent` 改为兼容壳层
- 保留当前 UI 事件流与结果结构
- 保证当前工作区交互不回退

### 阶段 5：校验与收尾

- 补充模块级单元测试
- 完成 `ruff check`
- 完成 `ruff format --check`
- 完成 `pytest`

## 当前 TODO

- [x] 抽离 Prompt 模板到独立模块
- [x] 引入 ToolRegistry 并迁移默认工具
- [x] 引入 Engine while-loop 运行时
- [x] 让 `PlanAndExecuteAgent` 委托到新 Engine
- [ ] 保持学术导读模式串行链路不回退
- [x] 补充新模块测试并完成全量校验

## 当前进展

- 已新增 `src/labflow/reasoning/agent_prompts.py`
  - 统一承载 Planner / Executor / RePlanner / Final Answer / Reflection / 学习助手 Prompt
- 已新增 `src/labflow/reasoning/agent_tools.py`
  - 引入 `AgentTool`、`ToolRegistry`、`ReasoningToolbox`
  - 默认工具已改为注册式管理
- 已新增 `src/labflow/reasoning/agent_engine.py`
  - 引入独立的 `PlanAndExecuteEngine`
  - 以清晰的 `while current_plan.steps` 驱动 Planner / Executor / RePlanner
- `PlanAndExecuteAgent` 已改为兼容壳层：
  - 现有 `aligner.py`、`ui/app.py` 入口不需要感知底层拆分
  - 实际运行时已切到新的 Engine / Prompt / Tool Registry
- 已新增运行时测试：
  - `tests/unit/test_agent_runtime.py`
  - 覆盖 ToolRegistry 与 Engine while-loop
- 当前全量校验已通过：
  - `ruff check app.py src tests`
  - `ruff format --check app.py src tests`
  - `pytest`

## 后续收尾点

- 下一轮继续把 `agent_executor.py` 里已经不再参与主执行链的旧 Planner / Executor / RePlanner 实现清理掉
- 继续下沉学术导读模式与结果后处理中的共性 Prompt / 规则，进一步缩小兼容壳层体积
