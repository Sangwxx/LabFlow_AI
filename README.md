# LabFlow AI

LabFlow AI 是一个面向实验室场景的论文-代码对齐分析工具。首版目标聚焦于：

- 上传文本型论文 PDF
- 导入本地 Git 仓库路径
- 提取论文与代码的结构化信息
- 生成可追溯的对齐分析结果

## 当前阶段

当前已完成阶段 0 初始化：

- 基础目录结构
- 环境变量与配置加载
- 最小化 Streamlit 首页
- lint / format / test 脚手架

## 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

## 启动页面

```bash
python -m streamlit run app.py
```

## 本地检查

```bash
ruff check .
ruff format --check .
pytest
```

## 目录结构

```text
.
├── app.py
├── src/
│   └── labflow/
│       ├── config/
│       ├── parsers/
│       ├── reasoning/
│       ├── reporting/
│       ├── retrieval/
│       └── ui/
├── tests/
│   ├── integration/
│   └── unit/
└── .agentdocs/
```

## 下一步

1. 实现 PDF 解析与章节抽取
2. 实现本地 Git 仓库结构解析
3. 建立 BM25 优先的轻量检索链路
4. 接入论文-代码对齐推理流程

