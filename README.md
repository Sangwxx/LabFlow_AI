# LabFlow

LabFlow 是一个面向论文阅读与源码定位的本地分析工具。应用支持上传论文 PDF、解析本地代码仓库，并在同一工作区中完成段落阅读、中文导读、术语解释与源码对齐分析。

## 核心能力

- 解析论文 PDF，并按可点击段落组织阅读视图
- 解析本地代码仓库结构，构建源码证据索引
- 基于多 Agent 编排完成论文段落导读与代码定位
- 在同一页面中展示论文上下文、推理链路与源码片段

## 技术栈

- Python 3.10+
- Streamlit
- PyMuPDF
- GitPython
- OpenAI-compatible API

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd Labflow
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制模板文件：

Windows:

```bash
copy .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

编辑 `.env`，填写可用的模型配置：

```env
APP_ENV=dev
API_KEY=
BASE_URL=https://api.qnaigc.com/v1
MODEL_NAME=moonshotai/kimi-k2.5
```

说明：

- `API_KEY` 为模型访问凭据
- `BASE_URL` 需要指向兼容 OpenAI Chat Completions 的服务端点
- `MODEL_NAME` 需要与对应服务支持的模型一致

### 5. 启动应用

推荐使用统一启动入口：

```bash
python run_labflow.py
```

也可以直接运行 Streamlit：

```bash
python -m streamlit run app.py --server.port 8501 --server.headless true
```

启动后默认访问：

- [http://127.0.0.1:8501](http://127.0.0.1:8501)

## Windows 一键启动

仓库提供了 Windows 启动脚本：

- [start_labflow.bat](/D:/Labflow/start_labflow.bat)

该脚本会自动完成以下步骤：

1. 创建本地虚拟环境 `.venv`
2. 安装运行依赖
3. 在缺少 `.env` 时从 `.env.example` 自动生成模板
4. 启动 Streamlit 服务

## Docker

### 构建镜像

```bash
docker build -t labflow .
```

### 启动容器

```bash
docker run --rm -p 8501:8501 --env-file .env labflow
```

容器默认监听：

- `0.0.0.0:8501`

### 挂载本地代码目录

如果需要在容器中分析宿主机上的代码仓库，需要显式挂载目录：

```bash
docker run --rm -p 8501:8501 --env-file .env -v /path/to/repos:/workspace/repos labflow
```

Windows PowerShell 示例：

```powershell
docker run --rm -p 8501:8501 --env-file .env -v E:\Projects:/workspace/repos labflow
```

此时页面中填写的代码目录路径应使用容器内路径，例如：

```text
/workspace/repos/VLN-DUET-main
```

## 使用方式

1. 启动应用并打开首页
2. 上传论文 PDF
3. 填写待分析代码仓库的本地路径
4. 进入工作区
5. 在左侧点击论文段落
6. 在右侧查看导读结果、术语解释、源码定位与推理链路

## 演示样例

为了方便比赛评委快速复现，我们在仓库里额外放了一套固定演示样例：

- 演示视频：[LabFlow 一体化科研助手（哔哩哔哩）](https://b23.tv/smNdUX4)
- `demo_assets/paper/Think.pdf`
- `demo_assets/code/VLN-DUET-main/`

其中：

- `demo_assets/paper/Think.pdf` 是测试用论文
- `demo_assets/code/VLN-DUET-main/` 是对应的代码项目快照，不包含原始 `.git` 历史

如果你只是想先看项目怎么展示，直接用这套样例即可。

## 目录结构

```text
.
├── app.py
├── run_labflow.py
├── start_labflow.bat
├── demo_assets/
│   ├── README.md
│   ├── paper/
│   │   └── Think.pdf
│   └── code/
│       └── VLN-DUET-main/
├── src/
│   └── labflow/
│       ├── clients/
│       ├── config/
│       ├── parsers/
│       ├── reasoning/
│       ├── reporting/
│       └── ui/
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── .env.example
```

## 开发与验证

安装开发依赖：

```bash
pip install -r requirements-dev.txt
```

执行检查：

```bash
ruff check .
ruff format --check .
pytest
```

## 运行说明

- 应用启动后，即使缺少 `API_KEY`，页面仍可打开
- 但论文导读、术语解释与代码对齐依赖模型服务，缺少可用凭据时无法正常完成推理
- 本项目当前主要面向本地使用场景；如果部署为公网服务，需要将“本地代码路径”替换为可上传或可拉取的代码输入形式
