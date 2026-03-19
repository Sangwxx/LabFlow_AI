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

## 风险与前置问题

### 已发现风险

- `.env` 中已有真实密钥，但当前仓库尚未见 `.gitignore`，需要在初始化时立即处理，避免误提交。
- 当前只拿到了比赛规则与七牛接口文档，尚未拿到用户对“演示重点”的明确偏好，可能影响 MVP 取舍。
- LangChain 被列为可用框架，但如果使用过深，容易模糊“自研 Agent 逻辑”的比赛边界，需要控制使用深度。
- ChromaDB 被列为建议依赖，但当前未确认可直接使用的远程 embedding 模型；如果首版保留向量检索，需改为本地 embedding 或先实现无向量的轻量检索兜底方案。

### 需要用户确认的问题

- 当前已确认，无新增阻塞问题。

## TODO

- [x] 读取根目录全部文件
- [x] 提炼比赛约束、技术栈与外部 API 能力
- [x] 初始化代理文档索引
- [x] 与用户确认 MVP 演示重点
- [x] 与用户确认 Git 仓库输入范围
- [x] 与用户确认是否将 OCR 纳入首版
- [x] 初始化项目骨架
- [x] 完成阶段 0 的基础工程搭建
