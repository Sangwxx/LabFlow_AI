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
- [x] 新增“源码导览”，把项目结构、相关模块职责与多实现线索统一展示出来
- [x] 新增“阅读笔记”，把已读论文片段与对应代码汇总成可下载的 Markdown 报告

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
  - 首页改为更紧凑的双卡片入口，隐藏首页侧边栏，避免数据输入与运行配置重复出现
  - 继续做了一轮结构整理：首页视图拆到 `ui/landing.py`，样式拆到 `ui/styles.py`，`app.py` 已压回 500 行以内
  - 右侧第二个 tab 现已升级为 `源码导览`，会先展示项目结构定位、再展示多个相关实现模块，最后落到主代码片段
- 更新 [D:\Labflow\README.md]
  - 改写为比赛交付导向的启动说明，明确评委快速启动方式、环境变量要求与 API Key 依赖
- 新增 [D:\Labflow\run_labflow.py]
  - 提供统一 Python 启动入口，缺少 `.env` 时自动复制模板
- 新增 [D:\Labflow\start_labflow.bat]
  - 提供 Windows 一键启动脚本，自动创建 `.venv`、安装依赖并启动页面
- 新增 [D:\Labflow\Dockerfile] 与 [D:\Labflow\.dockerignore]
  - 提供容器化运行入口，默认对外监听 `0.0.0.0:8501`
  - README 已补充容器挂载本地代码目录的说明，避免评委把宿主机路径直接填进容器页面
- 更新 [D:\Labflow\.env.example]
  - 补充字段说明，降低评委首次配置门槛
- 更新 [D:\Labflow\src\labflow\reasoning\code_grounding_agent.py]
  - 先看代码卡片，再做混合召回与调用链追踪
  - 为执行阶段增加运行时预算与兜底收束，避免真实联调时长期停在推理中
  - 为 `graph / map / global planning / coarse-scale` 相关段落增加焦点偏置与候选重排
  - 最终源码说明强制锚定当前展示片段，避免“解释说的是 A，UI 显示的是 B”
- 更新 [D:\Labflow\src\labflow\reasoning\agent_tools.py]
  - `llm_semantic_search` 改为基于本地代码卡片的语义召回
- 更新 [D:\Labflow\src\labflow\reasoning\evidence_builder.py]
  - Python 大文件切块不再只截文件头，改为保留分布式代表逻辑块，避免 `vilmodel.py` 后半段核心实现进不了索引
- 更新 [D:\Labflow\tests\unit\test_evidence_builder.py]
  - 增加代码卡片语义召回测试
  - 增加“大文件后半段逻辑块也要进入索引”的切块覆盖测试
- 更新 [D:\Labflow\tests\unit\test_agent_runtime.py]
  - 增加“执行器只返回兜底动作时提前收束”和“达到预算时生成超时计划”的测试
- 新增 [D:\Labflow\tests\unit\test_code_grounding_agent.py]
  - 增加图规划章节优先命中 `graph_utils.py` 与“解释必须锚定当前证据片段”的测试
  - 增加“跨模态编码器应优先命中编码层而不是图路径工具函数”的排序测试
- 更新 [D:\Labflow\tests\unit\test_ui_app.py]
  - 增加源码概览卡与代码 HTML 输出格式测试
- 更新 [D:\Labflow\src\labflow\reasoning\models.py]
  - 新增 `SourceGuideItem`，让源码导览能结构化承载相关实现模块
- 更新 [D:\Labflow\src\labflow\reasoning\code_grounding_agent.py]
  - 复用现有候选池与语义摘要，补出项目结构摘要和多模块导览信息
- 更新 [D:\Labflow\src\labflow\ui\styles.py]
  - 为源码导览新增项目结构块、模块卡片与概览样式
- 新增 [D:\Labflow\.agentdocs/frontend/architecture.md]、[D:\Labflow\.agentdocs/frontend/ui-design.md]、[D:\Labflow\.agentdocs/backend/architecture.md]
  - 补齐前后端治理文档，明确页面、推理层与测试约束

## 真实联调结论

- 在 `http://127.0.0.1:8501` 重新做了网页端实测，定位到一个真实回归：`PlanAndExecuteAgent._merge_learning_and_code_result()` 没把 `project_structure_context` 和 `source_guide` 透传回 UI，导致 `源码导览` 只剩主代码片段。
- 修复合并字段后，再把 `ALIGNMENT_CACHE_VERSION` 提升到 `learning-output-v16`，并重启 Streamlit 服务清掉旧的 `@st.cache_resource` Agent 实例；随后复测同一段 `3.2.2 Coarse-scale Cross-modal Encoder`，`源码导览` 已能稳定展示：
  - `项目结构定位`
  - `关联模块`
  - `相关实现`
  - `主代码片段`
- 当前网页端验证截图保存在 [D:\Labflow\labflow-source-guide-verified.png]，可直接用于后续演示视频取景。
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
- 进一步排查后确认根因之一是 `build_code_evidences()` 对 Python 文件默认只保留前 12 个逻辑块，导致 `map_nav_src/models/vilmodel.py` 后半段的 `GraphLXRTXLayer.forward / GlobalMapEncoder / forward_navigation_per_step` 长期缺席候选池
- 修复切块策略后，在新的 `http://127.0.0.1:8517` Playwright 复测里，同一段 `3.2.2 Coarse-scale Cross-modal Encoder` 的 `源码定位` 已稳定落到 `map_nav_src/models/vilmodel.py · L383-L398`，展示片段为 `GraphLXRTXLayer.forward`
- 后续在 `http://127.0.0.1:8520` 复测首页时，左侧重复侧边栏已移除，首页改为居中的双卡片入口，按钮与标题层级也更克制
- 结论：UI 可读性问题已基本收口，但代码对齐结果对同一论文段落仍存在跨轮次波动，下一轮需要继续压制 `agent / rollout / action` 这类泛化候选的回流
- 最新补充（`2026-03-22`）
  - `源码导览` 的概览卡片已从“算子核对 / 形状线索 / 人工复核”收敛为“这段代码在做什么 / 它和论文片段的关系”，减少无效解释。
  - `相关实现` 区块现在优先输出“这段代码具体做了什么”的说明，例如 `GraphMap.get_pos_fts` 会直接说明它在遍历候选节点并拼接相对位置特征，而不是继续重复抽象命中描述。
  - Playwright 已在重启后的 `http://127.0.0.1:8501` 复测确认：旧卡片不再显示，新的解释文案已经进入真实页面，可直接用于后续演示视频。
  - 根据产品取舍，`源码导览` 中原先单独铺开的 `【主代码片段】` 已删除；当前页面只保留“为什么命中 / 相关实现各自做什么”，避免用户把单一代码块误解为唯一答案。
  - 概览顶部补充的“这段代码在做什么 / 它和论文片段的关系”两张说明卡也已删除，避免在已有“关联模块 / 相关实现”解释之外继续重复同一层信息。
- 继续往前一轮后，代码结果区已经不再只返回单片段。当前 `源码导览` 会把项目主干目录、当前重点目录、多个相关实现模块和主代码片段统一放在同一个页签里，更适合视频演示“论文段落对应一组实现”的真实情况

## 下一轮实施约束

- `源码定位` 结果区需要升级为 `源码导览`，承认“一段论文对应多个代码实现”的现实，不再只展示单片段。
- 优先复用现有候选池、语义摘要与项目结构文本，不额外引入一条独立检索链。
- 导览区至少要覆盖三层信息：
  - 项目结构中的相关目录位置
  - 当前论文段落关联的多个实现模块及职责说明
  - 主命中代码片段与高亮行

## 后续关注点

- 侧边栏 `API Key` 已改为“空白输入 + 会话态保存覆盖值”，环境变量中的真实密钥不会再回显到页面
- 页面样式已移除 Google Fonts 外链，请求级 ORB 告警已消失
- 浏览器网络里仍能看到 `webhooks.fivetran.com` 的外联请求，代码仓内未检索到对应引用，建议下一轮确认是否来自 Streamlit 依赖、浏览器环境或外部注入
- 当前 `3.2.2 Coarse-scale Cross-modal Encoder` 这类图规划章节仍可能被 generic attention 逻辑块抢占候选前排，需要继续强化 `graph / map / global planning` 相关召回偏置

## 验证要求

- `ruff check app.py src tests`
- `ruff format --check app.py src tests`
- `pytest`

## 2026-03-23 补充

- 修补了 `CodeGroundingAgent` 里停在半途的源码解释优化，保留原有强命中规则的同时，新增基于 Python 函数结构的行为说明。
- 当前解释会优先覆盖三类容易说空话的场景：构造函数初始化、`save_to_json` 一类状态导出、`FloydGraph.path` 一类递归路径还原。
- 新增 `tests/unit/test_source_guide_summary.py` 的覆盖，确保源码导览摘要不再退化成“定义了 / 调用了”的模板句。
- 继续把“和当前论文片段的关系”改成证据句：优先输出“论文这里提到什么 + 代码负责什么 + 二者为何对应”，不再只给“图结构相关实现”这类空泛标签。
- 进一步修正源码导览排序：两三行的 helper / accessor 方法不再因为块很小就自动排到前面，`visited()`、`distance()` 这类碎片逻辑会在导览层被过滤掉。
- 补充 `tests/unit/test_code_grounding_agent.py` 覆盖，确保拓扑图章节的导览入口优先落到 `update_graph` 这类机制实现，而不是落到只返回布尔值的小函数。
- `ReasoningToolbox.read_code_segment` 现在接受执行器实际使用的别名参数：`file_path / start_line / end_line`，避免真实页面里追代码时误报“没有找到指定代码段”。
- 在 `http://127.0.0.1:8501` 的 Playwright 复测里，`3.1 Topological Mapping` 已能稳定把 `GraphMap.update_graph / GraphMap.get_pos_fts / FloydGraph.path` 作为源码导览主入口，不再退化成 `visited()` 这类两行 helper。
- 针对 `vilmodel.py` 的核心编码器方法补了专门摘要规则：`GlobalMapEncoder.gmap_input_embedding` 会直接解释“聚合视觉特征 + 叠加步编码与位置编码”，`GlocalTextPathNavCMT.forward_navigation_per_step` 会直接解释“全局分支 + 局部分支 + 动态融合”的单步导航决策流程。
- 将 `ALIGNMENT_CACHE_VERSION` 继续提升到 `learning-output-v21`，避免页面继续复用旧的源码导览缓存结果。
- 新增“阅读笔记” tab
  - 按会话累计已读论文片段和对应代码
  - 支持一键生成 Markdown 笔记并下载

## 2026-03-23 最小 RAG 改造补充

- 新增 [D:\Labflow\src\labflow\reasoning\code_knowledge_index.py]
  - 引入 `CodeKnowledgeIndex`，把代码切块、语义摘要、锚点词和轻量向量表示合并成统一的代码知识条目。
  - 初始召回从“词法 + 摘要 BM25”升级为“词法 BM25 + 轻量语义向量 + 结构偏置 + 可选 LLM rerank”的混合检索。
- 更新 [D:\Labflow\src\labflow\reasoning\code_grounding_agent.py]
  - `CodeGroundingAgent` 初始候选生成已接入 `CodeKnowledgeIndex`。
  - 对 `Topological Mapping` 增加显式机制约束：优先 `GraphMap.update_graph / GraphMap.get_pos_fts / FloydGraph.path` 这类建图主干，继续压低 `teacher_action / rollout / make_equiv_action` 这类任务层逻辑。
  - `源码导览` 的候选选择不再只看“小而具体”，而是优先更能代表论文机制的实现块。
- 更新 [D:\Labflow\src\labflow\reasoning\agent_tools.py]
  - `llm_semantic_search` 改为复用同一套代码知识索引，保证执行阶段与初始召回阶段使用一致的证据来源。
- 更新 [D:\Labflow\src\labflow\reasoning\evidence_builder.py]
  - 语义摘要阶段改为更稳健的异常回退，测试里的假 LLM 或非结构化响应不会再中断整条对齐链。
- 新增 [D:\Labflow\tests\unit\test_code_knowledge_index.py]
  - 覆盖机制块优先于 trivial helper、知识索引可选 LLM rerank、以及工具层语义搜索复用知识索引三类行为。
- 更新 [D:\Labflow\tests\unit\test_code_grounding_agent.py]
  - 新增 `Topological Mapping` 场景下 `teacher_action` 不应污染源码导览前排的回归测试。
- 网页端实测结论
  - 在 [http://127.0.0.1:8501](http://127.0.0.1:8501) 上，`3.1 Topological Mapping` 已稳定收敛到：
    - `GraphMap.update_graph`
    - `GraphMap.get_pos_fts`
    - `FloydGraph.path`
  - 这轮最关键的回归已经被打掉：`GMapNavAgent._teacher_action_r4r` 不再出现在源码导览第一位。
  - `3.2.2 Coarse-scale Cross-modal Encoder` 的本地逻辑已能产出：
    - `GlocalTextPathNavCMT.forward_navigation_per_step`
    - `GlobalMapEncoder.gmap_input_embedding`
    - `GraphLXRTXLayer.forward`
    这组机制级实现；但网页端该片段当前推理耗时仍偏长，演示前还需要再压一次运行时收敛。
- 当前剩余问题
  - `Coarse-scale Cross-modal Encoder` 页面结果虽然机制命中已对，但真实网页交互下偶尔要等待较久，影响演示稳定性。
  - `代码入口` 仍是压缩的一行代码预览，可读性一般，后续可以改成真正的多行代码视图。
