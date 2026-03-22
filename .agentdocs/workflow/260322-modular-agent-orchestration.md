# 任务文档：重构为 Planner + 多子 Agent 协作链

## 任务目标

把当前“翻译、导读、术语、代码对齐”混在一起的推理链，重构为由总控 Planner 决定启用哪些子 Agent 的模块化编排：

1. `TranslationAgent`
2. `ReadingAgent`
3. `CodeGroundingAgent`
4. 顶层 `PlanAndExecuteAgent` 只负责规划和编排

同时修复“代码解析功能被压制”的问题，让真正有价值的源码候选能够在 UI 中展示出来。

## 当前问题

- 章节预审过于保守，很多片段直接被切去学术导读，代码链没有机会启动。
- 代码对齐拒绝策略过重，弱到中等可信的源码候选被直接吞掉。
- UI 仅展示强匹配源码，导致“找到了可疑实现，但看不到”的体验。
- `agent_executor.py` 过长，已经超过单文件复杂度上限。

## 方案概述

### 总体结构

```text
reasoning/
├── agent_engine.py          # Planner / Executor / RePlanner while-loop
├── agent_prompts.py         # Planner / Executor / RePlanner Prompt
├── agent_tools.py           # Tool Registry
├── learning_agents.py       # TranslationAgent + ReadingAgent
├── code_grounding_agent.py  # CodeGroundingAgent
├── agent_executor.py        # 顶层编排入口
├── evidence_builder.py      # 证据构建
└── models.py                # 数据模型
```

### 关键策略

- Planner 输出 `section_type + enabled_agents + code_focus + steps`
- Translation / Reading / Code Grounding 三条链彼此解耦
- `CodeGroundingAgent` 只负责源码候选召回、ReAct 执行和最终代码解释
- UI 放宽源码展示阈值，只要已经拿到可信实现解释就展示，不再死锁在 strong match

## TODO

- [x] 为 Planner 增加 `enabled_agents`、`section_type`、`code_focus`
- [x] 拆出 `TranslationAgent`
- [x] 拆出 `ReadingAgent`
- [x] 拆出 `CodeGroundingAgent`
- [x] 让顶层 Agent 只负责编排，不再自己塞满所有细节
- [x] 放宽源码展示阈值，恢复代码联动可见性
- [x] 把 `agent_executor.py` 压回 1000 行以内
- [x] 补充测试覆盖新的 Planner / 子 Agent / UI 行为
- [x] 接入本地代码卡片与混合召回，增强 `CodeGroundingAgent` 的源码定位能力
- [x] 跟进一轮真实仓库联调，继续调优 `code_focus` 与召回质量
- [x] 修正“源码说明与实际展示片段错位”的结果拼装问题
- [x] 把右侧结果区改成 tab 布局，避免源码区域被长导读内容挤到下方
- [x] 优化代码卡片可读性，解决长代码行截断与空白区域过大的问题
- [ ] 继续压低代码命中结果的波动，避免同一论文段落在不同轮次漂到 `agent_obj.py` 等泛化候选

## 当前进展

- 新增 [D:\Labflow\src\labflow\reasoning\learning_agents.py]
  - 承载翻译与导读链
- 新增 [D:\Labflow\src\labflow\reasoning\code_grounding_agent.py]
  - 承载源码对齐链
- 重写 [D:\Labflow\src\labflow\reasoning\agent_executor.py]
  - 改成 Planner 驱动的模块编排入口
- 更新 [D:\Labflow\src\labflow\reasoning\agent_engine.py]
  - Planner 现在能输出启用模块与源码焦点
- 更新 [D:\Labflow\src\labflow\reasoning\agent_prompts.py]
  - Prompt 显式约束 Planner 做模块决策
- 更新 [D:\Labflow\src\labflow\ui\app.py]
  - 源码展示不再只限于强匹配
  - 右侧结果区改为 `导师讲解 / 源码定位 / 推理链路` tab
  - `源码定位` 改为“概览在上、代码整块铺开”的结构，避免代码被长导读挤到下方
  - 修复代码块 HTML 被当作原始文本输出的问题，代码行现在按真实代码渲染并自动换行
- 更新 [D:\Labflow\src\labflow\reasoning\code_grounding_agent.py]
  - 先看代码卡片，再做混合召回与调用链追踪
  - 为执行阶段增加运行时预算与兜底收束，避免真实联调时长期停在推理中
  - 为 `graph / map / global planning / coarse-scale` 相关段落增加焦点偏置与候选重排
  - 最终源码说明强制锚定当前展示片段，避免“解释说的是 A，UI 显示的是 B”
- 更新 [D:\Labflow\src\labflow\reasoning\agent_tools.py]
  - `llm_semantic_search` 改为基于本地代码卡片的语义召回
- 更新 [D:\Labflow\tests\unit\test_evidence_builder.py]
  - 增加代码卡片语义召回测试
- 更新 [D:\Labflow\tests\unit\test_agent_runtime.py]
  - 增加“执行器只返回兜底动作时提前收束”和“达到预算时生成超时计划”的测试
- 新增 [D:\Labflow\tests\unit\test_code_grounding_agent.py]
  - 增加图规划章节优先命中 `graph_utils.py` 与“解释必须锚定当前证据片段”的测试
- 更新 [D:\Labflow\tests\unit\test_ui_app.py]
  - 增加源码概览卡与代码 HTML 输出格式测试

## 真实联调结论

- 已用 Playwright MCP 在 `http://127.0.0.1:8511` 联调 `E:\文献\think.pdf` 与 `E:\VLN-DUET-main`
- 选中第 4 页 `3.2.2 Coarse-scale Cross-modal Encoder` 后，右侧结果区现在可以正常落出：
  - `【中文译文】`
  - `【核心要点】`
  - `【术语百科】`
  - `【源码落地】`
- 本轮真实联调命中的源码候选为 `pretrain_src/model/vilmodel.py · L79-L141`，页面代码卡片与解释区渲染正常
- 这次调优后，主问题从“长时间停留在 Agent 正在推理”收敛为“在预算内输出兜底但可读的结果”，避免 UI 看起来卡死
- 后续在 `http://127.0.0.1:8513` 继续联调同一段落时，源码定位已能稳定落到 `map_nav_src/models/graph_utils.py · L95-L168`，并且右侧结果区切成 tab 后，代码不再需要滚到很下方才能看到
- 修复代码块 HTML 渲染错误后，`源码定位` 页签不再把 `<div>/<span>` 原样打印到页面，代码行可正常查看
- 最新在 `http://127.0.0.1:8515` 用最终版布局复测时，界面已经切到“概览在上、代码整块铺开”的结构，但源码命中结果又漂到了 `map_nav_src/reverie/agent_obj.py · L30-L498`
- 结论：UI 可读性问题已基本收口，但代码对齐结果对同一论文段落仍存在跨轮次波动，下一轮需要继续压制 `agent / rollout / action` 这类泛化候选的回流

## 后续关注点

- 侧边栏 `API Key` 已改为“空白输入 + 会话态保存覆盖值”，环境变量中的真实密钥不会再回显到页面
- 页面样式已移除 Google Fonts 外链，请求级 ORB 告警已消失
- 浏览器网络里仍能看到 `webhooks.fivetran.com` 的外联请求，代码仓内未检索到对应引用，建议下一轮确认是否来自 Streamlit 依赖、浏览器环境或外部注入
- 当前 `3.2.2 Coarse-scale Cross-modal Encoder` 这类图规划章节仍可能被 generic attention 逻辑块抢占候选前排，需要继续强化 `graph / map / global planning` 相关召回偏置

## 验证要求

- `ruff check app.py src tests`
- `ruff format --check app.py src tests`
- `pytest`
