# 任务文档：项目初始化与建项规划

## 任务目标

在不直接开始大规模编码的前提下，先完成以下工作：

1. 读取仓库根目录全部文件并提炼真实约束。
2. 明确本项目作为华科 AI 大赛参赛作品的产品定位、技术边界与合规要求。
3. 输出可执行的分阶段实施计划。
4. 在关键信息不明确时，先向用户确认，再初始化项目。

## 当前仓库状态

- 当前仓库已完成阶段 0 初始化，已具备最小可运行项目骨架。
- 当前根目录核心文件包括：
  - `.env`
  - `.env.example`
  - `.gitignore`
  - `README.md`
  - `requirements.txt`
  - `requirements-dev.txt`
  - `ruff.toml`
  - `pytest.ini`
  - `app.py`
  - `.agentdocs/`
  - `src/`
  - `tests/`
- 已确认 `.env` 中存在至少以下运行配置：
  - `API_KEY`
  - `BASE_URL`
  - `MODEL_NAME`
- Git 状态：
  - 已执行 `git init`
  - 已创建 `.gitignore`，并显式忽略 `.env` 与 `.codexrules`

## 已确认约束

### 比赛与工程约束

- 必须自研 Agent 逻辑，不能将第三方 Agent 平台成品直接拿来参赛。
- 需要保留“个人开发痕迹”，禁止在代码、注释、文档、提交信息中暴露 AI 辅助信息。
- 每个子功能应独立提交，避免一次性提交大量代码。
- 代码需要模块化，并体现多人协作下的接口边界、测试意识与变更说明。

### 技术栈约束

- Python 3.10+
- Streamlit：作为首选界面框架
- PyMuPDF：用于 PDF 解析
- LangChain：可作为 Agent 编排框架，但核心推理逻辑仍需本项目自研
- ChromaDB：向量存储
- GitPython：Git 仓库解析
- OpenAI 兼容调用方式：通过七牛提供的兼容端点接入模型

### 外部能力边界

- 七牛兼容模型推理接口：`/v1/chat/completions`
- 七牛模型列表接口：`/v1/models`
- 七牛 OCR：`/v1/ocr`
- 七牛全网搜索接口：`https://ai.qiniuapi.com/v1/search`
- 七牛 MCP 接入能力：可作为后续扩展能力，不应成为首版建项阻塞项

### 已做接口核验

- 已实测 `BASE_URL/models` 可访问，说明当前 `.env` 中的七牛推理配置可用于项目初始化阶段。
- 当前模型列表中已确认存在多种通用推理模型，例如 `moonshotai/kimi-k2.5`、`openai/gpt-5.4`、`openai/gpt-5.4-mini`、`qwen3-coder-480b-a35b-instruct` 等。
- 当前模型列表中未检索到名称包含 `embed` 或 `embedding` 的模型，首版不要把“远程 embedding 模型”当作默认前提。

## 产品理解

当前最合理的产品抽象是“围绕一篇论文与一个代码仓库建立可追溯对齐关系，并产出实验室协作用分析结果”的单人 Web 应用。

建议将首版目标压缩为一个清晰 MVP：

1. 输入论文 PDF。
2. 输入或选择本地 Git 仓库路径。
3. 提取论文结构化内容与代码仓库结构化内容。
4. 建立章节/方法/模块之间的关联候选。
5. 生成一份可读的对齐分析与审计周报。

## 推荐架构

## 目录建议

```text
labflow-ai/
├── src/
│   ├── app/
│   │   └── main.py
│   ├── config/
│   │   └── settings.py
│   ├── clients/
│   │   └── llm_client.py
│   ├── parsers/
│   │   ├── pdf_parser.py
│   │   └── git_parser.py
│   ├── indexing/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   ├── reasoning/
│   │   ├── aligner.py
│   │   ├── analyzer.py
│   │   └── report_builder.py
│   ├── services/
│   │   └── audit_service.py
│   └── ui/
│       ├── app.py
│       └── sections/
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
├── config/
├── requirements.txt
└── README.md
```

### 关键设计原则

- 先做“单机单用户可跑通”的本地应用，不在首版引入后端服务拆分。
- 将“解析”“检索”“推理”“报告输出”四条链路解耦，便于比赛演示时单独展示每一层能力。
- LLM 调用统一封装到 `clients/llm_client.py`，避免业务层散落接口细节。
- 向量存储与检索逻辑单独抽象，后续替换 ChromaDB 实现时不影响上层。
- UI 仅负责输入、过程展示和结果呈现，不混入复杂业务逻辑。

## 分阶段实施计划

### 阶段 0：建项准备

- 初始化目录结构、依赖声明、环境变量加载、`.gitignore`
- 建立基础配置对象与统一异常处理
- 搭建最小可运行 Streamlit 首页

### 阶段 1：感知层

- 实现 PDF 文本提取与章节结构识别
- 实现 Git 仓库遍历、文件过滤、模块摘要与提交信息读取
- 为两类输入构建统一的中间表示

### 阶段 2：索引层

- 文本切分与元数据附着
- 嵌入生成与 ChromaDB 写入
- 基于任务类型的检索接口封装

### 阶段 3：推理层

- 实现论文章节到代码模块的候选匹配
- 实现依据证据片段的可解释分析
- 区分“强匹配”“弱匹配”“缺失实现”“疑似偏离”几类结果

### 阶段 4：行动层

- 生成审计周报/实验室协作报告
- 支持导出 Markdown 文本
- 在 UI 中展示证据链与结论摘要

### 阶段 5：增强能力

- 接入七牛搜索用于补充论文背景或依赖资料检索
- 接入 OCR 处理扫描版 PDF
- 评估 MCP 作为后续工具编排扩展点

## 首版建议范围

为保证比赛可交付性，首版建议只做以下闭环：

- 本地上传 PDF
- 输入本地 Git 仓库路径
- 解析并抽取结构化信息
- 调用一个统一的 LLM 推理链完成章节-模块对齐
- 输出 Markdown 审计报告
- 用 Streamlit 展示过程和结果

暂不建议首版直接加入：

- 多用户系统
- 登录鉴权
- 在线代码仓库拉取
- 复杂工作流编排
- 长链路多 Agent 协作
- 数据库持久化服务拆分

## 已确认的产品决策

- 首版演示重点：论文-代码对齐分析
- Git 导入方式：本地路径导入
- PDF 范围：仅支持文本型 PDF
- 检索兜底方案：优先使用本地 BM25，后续再视情况接入本地 embedding

## 阶段 0 完成情况

- 已初始化 Git 仓库
- 已完成第一次提交：`chore: 初始化项目骨架与 Streamlit 首页`
- 已创建基础目录结构与 `src` 目录布局
- 已完成配置加载、环境变量样例与根目录启动入口
- 已完成最小化 Streamlit 首页
- 已补充单元测试与集成测试脚手架
- 已完成本地校验：
  - `ruff check app.py src tests`
  - `ruff format --check app.py src tests`
  - `pytest`
  - `streamlit run app.py --server.headless true`

## 阶段 1 当前进展

- 已实现 `PDFParser`，支持：
  - 本地文件解析
  - 上传字节流解析
  - 标题/正文块启发式区分
  - PDF 路径不存在、空文件、加密文档等基础异常提示
- 已实现 `GitRepoParser`，支持：
  - 本地路径或仓库子目录定位仓库根
  - 提取最近 10 次提交
  - 提取当前工作区相对 `HEAD` 的 diff
  - 路径不存在、非仓库路径等基础异常提示
- 已在首页接入侧边栏：
  - PDF 上传组件
  - 本地 Git 路径输入框
- 已完成阶段 1 当前范围内的本地校验：
  - `ruff check app.py src tests`
  - `ruff format --check app.py src tests`
  - `pytest`
  - `streamlit run app.py --server.headless true`

## 阶段 2 当前进展

- 已实现推理层数据模型：
  - `PaperSection`
  - `CodeEvidence`
  - `AlignmentCandidate`
  - `AlignmentResult`
- 已实现 `EvidenceBuilder`，支持：
  - 从 PDF 结构块重组章节
  - 从 Git diff 拆分文件级证据
  - 基于 BM25 风格评分召回候选章节
- 已实现 `PaperCodeAligner` 与模块级 `align()` / `align_inputs()` 入口
- 已补充统一 `LLMClient`，通过 OpenAI 兼容接口请求模型并解析 JSON 结果
- 已在首页接入推理层：
  - 真实输入对齐按钮
  - 内置参数错配案例展示
- 已完成阶段 2 当前范围内的本地校验：
  - `ruff check app.py src tests`
  - `ruff format --check app.py src tests`
  - `pytest`
  - `streamlit run app.py --server.headless true`

## 阶段 3 当前进展

- 已实现 `ReportGenerator`，支持：
  - 汇总项目概况
  - 输出高风险错配项与一致性良好项
  - 汇总改进建议
  - 导出标准 Markdown 审计周报
- 已增强首页可视化：
  - 风险预警仪表盘
  - 总体对齐置信度 `st.metric`
  - 评分低于 6 分的可疑错配项置顶
  - Markdown 报告下载按钮
- 已补充结果缓存逻辑：
  - 真实输入分析结果优先展示
  - 无真实结果时回退到内置参数错配案例
- 已完成阶段 3 当前范围内的本地校验：
  - `ruff check app.py src tests`
  - `ruff format --check app.py src tests`
  - `pytest`
  - `streamlit run app.py --server.headless true`

## 联调阶段紧急修复

- 已完成 Streamlit 首页重构：
  - 主界面改为 `数据采集 / 分析报告 / 历史记录` 三标签布局
  - 所有输入配置统一收纳到侧边栏
  - 未执行分析前主界面仅保留欢迎语、状态摘要与操作指南
  - 分析完成后尝试自动切换到“分析报告”标签页
- 已修复 PDF 上传流兼容性：
  - `PDFParser` 现支持本地路径、裸 `bytes`、`BytesIO` 和上传组件类文件对象
  - 缺少 `PyMuPDF` 时不再把页面打崩，而是在 UI 中给出显式安装提示
- 已增强普通源码目录容错：
  - `GitRepoParser` 在无 `.git` 时回退为“普通源码目录”模式
  - 会将目录中的文本源码文件包装为快照式 diff，继续进入 BM25 召回与推理链路
  - 不再因 GitHub Zip 解压目录直接报错中断
- 已增强报告页联调可观测性：
  - 保留风险预警仪表盘
  - 置顶展示评分低于 6 分的可疑项
  - 增加 BM25 召回观察面板，便于判断召回章节是否与代码目录相关
  - 保留 Markdown 报告下载入口
- 已完成本轮紧急修复的本地校验：
  - `.\.tools\ruff\ruff.exe check app.py src tests`
  - `.\.tools\ruff\ruff.exe format --check app.py src tests`
  - `python -m pytest`

## 风险与前置问题

### 已发现风险

- `.env` 中已有真实密钥，但当前仓库尚未见 `.gitignore`，需要在初始化时立即处理，避免误提交。
- 当前只拿到了比赛规则与七牛接口文档，尚未拿到用户对“演示重点”的明确偏好，可能影响 MVP 取舍。
- LangChain 被列为可用框架，但如果使用过深，容易模糊“自研 Agent 逻辑”的比赛边界，需要控制使用深度。
- ChromaDB 被列为建议依赖，但当前未确认可直接使用的远程 embedding 模型；如果首版保留向量检索，需改为本地 embedding 或先实现无向量的轻量检索兜底方案。

### 需要用户确认的问题

- 当前已确认，无新增阻塞问题。

## 当前补充约束

- 联调期页面不得在每次重绘时自动触发 PDF/Git 解析，必须改为显式按钮触发，避免主界面被异常提示淹没。
- 代码目录输入必须兼容“真实 Git 仓库”和“Zip 解压后的普通源码目录”两类场景。
- PDF 上传必须优先支持 Streamlit 上传组件返回的内存流对象，不能只假设输入是本地路径。

## 当前交互重构

- 已将首页交互从“报告输出优先”切换为“知云式联动阅读优先”：
  - 主界面采用 `st.columns([2, 1])` 双栏布局
  - 左栏负责 PDF 预览与章节选择
  - 右栏负责展示与当前章节最相关的代码片段
- 已引入浏览器原生 PDF 预览组件封装：
  - 上传后的 PDF 字节流可直接在页面左侧显示
  - 不依赖解析成功即可完成文档预览
  - 若已选中章节，会优先把预览聚焦到对应页码
- 已将代码侧联动逻辑切换为“局部检索”：
  - 点击左侧章节目录后，右栏仅针对该章节执行局部召回
  - 右栏输出真实代码片段、文件路径、行号范围与局部对齐评分
  - 不再默认执行全量报告分析
- 已增强代码源降级策略：
  - 对真实 Git 仓库，会扫描仓库内 `.py` 文件构建代码片段
  - 对无 `.git` 的目录，会直接扫描 `.py` 文件作为代码源
  - 代码证据切片按函数/类边界优先拆分，尽量保证右栏阅读连续性
- 已完成本轮“知云模式”改造的本地校验：
  - `.\.tools\ruff\ruff.exe check app.py src tests`
  - `.\.tools\ruff\ruff.exe format --check app.py src tests`
  - `python -m pytest`
  - `streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501`

## 当前路由重构

- 已将主页面进一步拆成“门户入口 -> 阅读工作区”两段式路由：
  - 首页仅保留欢迎文案、PDF 上传入口、代码路径入口和“进入工作区”按钮
  - 工作区独占主区域，不再展示路径/API 等配置表单
  - 配置项统一沉到折叠侧边栏
- 已将工作区布局进一步收敛为沉浸式阅读视图：
  - 左栏按 `st.columns([1.5, 1])` 展示章节定位器和全高 PDF 预览
  - 右栏只保留局部代码片段和对齐结论
  - 不再展示召回调试面板、仪表盘或中间态摘要
- 已补充当前界面的纯逻辑测试：
  - 即时对齐结论评分与文案生成已纳入单元测试

## 当前布局微调

- 已将工作区代码侧继续压缩为“单条结论提示 + 固定高代码块”两层结构，避免右栏被说明文字吞掉。
- 已通过全局 CSS 收紧主容器边距、列宽约束与代码行高，优先追求更接近 IDE 的占屏效果。
- 当前工作区列宽继续采用 `st.columns([1.5, 1])`，在保证 PDF 可读性的同时尽量放大代码区的有效宽度。

## 当前语义对齐重构

- 已将推理链从“BM25 主导”切换为“LLM 语义理解主导”：
  - BM25 仅保留为候选裁剪与首轮召回
  - `align_section()` 会把论文片段与多段代码候选一起交给模型做语义选择
  - 模型需要基于算法结构、控制流程、张量流向与模块职责做判断，而不是仅依赖变量名相似度
- 已扩展结构化对齐结果：
  - `AlignmentResult` 新增 `semantic_evidence`
  - `AlignmentResult` 新增 `highlighted_line_numbers`
  - 右侧代码面板直接消费模型给出的关键逻辑行进行高亮
- 已将工作区右栏改为“Agent 理解结论 + Agent 理解证据 + 逻辑行高亮代码”的语义阅读模式：
  - 点击论文片段后，右栏优先展示模型解释为什么这段代码实现了论文机制
  - 命中的关键实现行会在代码画布中高亮，降低人工二次定位成本
- 已补齐本轮语义重构的单元测试：
  - 对齐器测试覆盖 `semantic_evidence` 与 `highlighted_lines`
  - UI 测试覆盖语义代码高亮 HTML 输出

## 当前 Agent 架构重构

- 已将线性 `aligner.py -> 单次裁决` 改造为 `agent_executor.py -> ReAct 循环`：
  - 第一步先生成“检索计划”
  - 若首轮证据不足，自动沿函数名、类名和关键变量继续追踪调用链
  - 在进入最终结论前，保留每轮“观察”结果，形成实现链路上下文
- 已为代码侧新增“语义索引”能力：
  - `EvidenceBuilder` 现在可以对仓库代码片段生成 `CodeSemanticSummary`
  - 每个索引条目包含逻辑摘要、职责、定义符号、调用符号和锚点词
  - 首轮召回不再只看原始代码切片，而是优先看语义摘要
- 已为输出层新增“反思层”：
  - 最终结果进入 UI 前会再经过一轮自我审计
  - 若置信度低于 8 分，强制提示“建议人工核对”
  - 结构化结果已新增检索计划、实现链路分析、自我审计和人工核对标记
- 已将工作区右栏升级为 Agent 结果面板：
  - 展示检索计划
  - 展示实现链路分析
  - 展示 Agent 理解证据
  - 展示自我审计与低置信度提醒
- 已完成本轮 Agent 改造的本地校验：
  - `.\.tools\ruff\ruff.exe check app.py src tests`
  - `.\.tools\ruff\ruff.exe format --check app.py src tests`
  - `python -m pytest`

## 当前 Plan-and-Execute 重构

- 已参考 Planner / Executor / RePlanner 的任务拆解方式，将控制中心正式收敛为 `PlanAndExecuteAgent`
- 当前执行链路变为：
  - `Planner.create_plan()` 先根据 `PaperSection` 和项目结构生成多步计划
  - `Executor.execute()` 针对单个步骤使用工具箱做 ReAct 式执行
  - `RePlanner.update_plan()` 根据最新 Observation 决定是否继续执行后续步骤
  - 最终再进入总结器与反思器输出实现链路分析
- 当前工具箱已具备：
  - `list_project_structure()`
  - `read_code_segment(path, line_start, line_end)`
  - `llm_semantic_search(query)`
- 当前 UI 已在右侧区域接入执行态可视化：
  - `st.status` 实时展示 `[Current Plan]`
  - `st.status` 实时展示 `[Thought]`
  - `st.status` 实时展示 `[Action]`
  - 最终通过 `st.expander` 保留完整执行轨迹，便于复盘 Agent 过程
- 当前结构上，`aligner.py` 已降为兼容层：
  - 保留既有 `align()` / `align_inputs()` / `align_section()` 入口
  - 实际执行逻辑统一委托给 `PlanAndExecuteAgent`

## 当前按需 ReAct 重构

- 已停止所有“启动即全量索引”的逻辑：
  - `app.py` 初始化阶段只准备 PDF 结构和项目文件树
  - 不再在启动或进入工作区时提前调用任何全量语义索引
  - 代码证据改为只有点击具体 `PaperSection` 后才懒加载
- 已将 Agent 执行机制收敛为 `Thought -> Action -> Observation -> Final Answer`：
  - Planner 先根据论文片段和文件树拆计划
  - Executor 再按需调用 `list_project_structure / read_code_segment / llm_semantic_search`
  - RePlanner 根据 Observation 决定是否继续下一轮
- 已修复 `EvidenceBuilder._summarize_code_evidence()` 的坏响应防御：
  - 当 LLM 返回非 `dict` 时不再触发 `payload.get(...)` 崩溃
  - 现在会稳定回退到本地结构化摘要
- 已增强工作区右栏的 Agent 过程可视化：
  - 使用 `st.status("Agent 正在推理...")` 展示 `[Current Plan] / [Thought] / [Action] / [Observation]`
  - 代码画布保持 `white-space: pre` 与横向滚动
  - 主容器宽度已收紧到 `max-width: 98vw`
- 本轮重构已完成本地校验：
  - `.\.tools\ruff\ruff.exe check app.py src tests`
  - `.\.tools\ruff\ruff.exe format --check app.py src tests`
  - `python -m pytest`

## 当前稳定性修复

- 已收口 LLM 请求失败路径：
  - `LLMClient.generate_json()` 在 `429 rate limit` 场景下会短退避重试
  - 若仍失败，则统一回退为 `None`，由上层 Agent 走本地兜底逻辑
  - 不再把 OpenAI SDK 异常直接抛到 Streamlit 页面顶层
- 已增强脏 JSON 兼容：
  - 会自动剥离代码块包裹
  - 会清洗非法控制字符
  - 会尝试从混杂文本里裁出最外层 JSON 对象
  - 若仍无法解析，则回退为 `None`
- 已补充回归测试：
  - `test_llm_client.py` 覆盖代码块 JSON、控制字符清洗和无效 JSON 回退
  - `test_aligner.py` 覆盖模型完全不可用时的 Agent 降级
- 已完成本轮本地校验：
  - `.\.tools\ruff\ruff.exe check app.py src tests`
  - `.\.tools\ruff\ruff.exe format --check app.py src tests`
  - `python -m pytest`
  - `streamlit` 本地冒烟探活返回 `200`

## 当前导师式交互整改

- 已将 Agent 的角色定位从“编译器式变量核对”调整为“导师式科研解释”：
  - System Prompt 明确要求优先解释算法思想、理论动机与工程落点
  - 当本地未命中可靠代码时，必须自动切换为“学术解释模式”
  - Final Answer 必须覆盖“逻辑对齐”和“科研补完”两部分
- 已将工作区右栏改为“结果优先、链路折叠”的展示模式：
  - 推理过程中通过 `st.status` 实时展示 `Current Plan / Thought / Action / Observation`
  - 推理完成后默认仅保留最终结论、科研补完、自我审计与代码高亮
  - 中间推理过程统一收进“查看推理链路”折叠面板，降低 UI 噪音
- 已强化按需 ReAct 的优雅降级：
  - 模型未稳定返回结构化动作时，Executor 会结束当前步骤并沿已有证据继续推进
  - 模型未稳定返回 Final Answer 时，Agent 会给出本地兜底的实现入口判断
  - 若没有代码证据，则返回论文阅读助手式解释，而不是直接断流
- 已补齐本轮本地校验：
  - `.\.tools\ruff\ruff.exe check app.py src tests`
  - `.\.tools\ruff\ruff.exe format --check app.py src tests`
  - `python -m pytest`
  - `streamlit` 本地冒烟探活返回 `200`

## TODO

- [x] 读取根目录全部文件
- [x] 提炼比赛约束、技术栈与外部 API 能力
- [x] 初始化代理文档索引
- [x] 与用户确认 MVP 演示重点
- [x] 与用户确认 Git 仓库输入范围
- [x] 与用户确认是否将 OCR 纳入首版
- [x] 初始化项目骨架
- [x] 完成阶段 0 的基础工程搭建
