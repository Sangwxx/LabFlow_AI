# LabFlow AI - 项目规范文档

> 创建时间: 2026-03-19  
> 文档版本: v1.0

---

## 1. 项目身份 (Project Identity)

### 1.1 项目名称
**LabFlow AI** —— 论文代码关联阅读与实验室协作助手

### 1.2 项目目标

通过 AI Agent 技术，解决课题组内「论文理论」与「代码工程」脱钩的问题。

**核心能力闭环**:
- **感知 (Perceive)**: 解析 PDF 与 Git 仓库
- **推理 (Reason)**: 将代码实现与论文理论对齐
- **行动 (Act)**: 生成审计周报

辅助实验室工作，实现论文与代码的智能关联分析。

### 1.3 核心模式

```
Perceive (感知) -> Reason (推理) -> Act (执行)
```

---

## 2. 技术栈约束 (Tech Stack)

### 2.1 编程语言
- **Python 3.10+**

### 2.2 核心依赖库

| 库名称 | 用途 | 版本要求 |
|--------|------|----------|
| Streamlit | UI 界面 | 最新稳定版 |
| PyMuPDF | PDF 解析 | 最新稳定版 |
| LangChain | Agent 框架 | 最新稳定版 |
| ChromaDB | 向量存储 | 最新稳定版 |
| GitPython | Git 交互 | 最新稳定版 |

### 2.3 API 风格约束

**所有 LLM 调用必须符合 OpenAI 兼容格式**。

```python
# 标准调用示例
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.qnaigc.com/v1"  # 或其他兼容端点
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}]
)
```

---

## 3. 大赛硬红线 (Competition Constraints)

### 3.1 严禁黑盒

❌ **禁止**直接调用第三方现成的 Agent 平台（如 Coze/Dify 导出的成品）。

✅ **必须**自研 Agent 逻辑，核心推理流程需自行实现。

### 3.2 工程痕迹

- 每完成一个子功能，**必须**提示用户进行 Git Commit
- **严禁**一次性提交数千行代码
- 保持清晰的开发迭代记录

**提交规范示例**:
```bash
# ✅ 正确的提交方式
git commit -m "feat: 实现 PDF 文本提取模块"
git commit -m "feat: 添加 Git 仓库解析功能"
git commit -m "fix: 修复向量存储索引问题"

# ❌ 错误的提交方式
git commit -m "feat: 完成全部功能"  # 一次性提交大量代码
```

### 3.3 多人协作模拟

即便单人参赛，也请在代码中保持模块化，遵循 Pull Request 的逻辑：

- 功能模块化设计
- 清晰的接口定义
- 独立的测试用例
- 详细的变更说明

---

## 4. 编码风格 (Coding Standards)

### 4.1 目录结构

```
labflow-ai/
├── src/
│   ├── parsers/          # 解析器模块
│   │   ├── pdf_parser.py
│   │   └── git_parser.py
│   ├── engine/           # 对齐逻辑模块
│   │   ├── aligner.py
│   │   └── analyzer.py
│   ├── ui/               # 界面模块
│   │   └── app.py
│   └── utils/            # 通用工具模块
│       └── helpers.py
├── tests/                # 测试目录
├── docs/                 # 文档目录
├── config/               # 配置文件
├── requirements.txt
└── README.md
```

### 4.2 文档化要求

**每个函数必须包含 docstring**，说明输入输出和逻辑。

```python
def extract_pdf_text(pdf_path: str, page_range: tuple = None) -> dict:
    """
    从 PDF 文件中提取文本内容。
    
    Args:
        pdf_path (str): PDF 文件的本地路径或 URL
        page_range (tuple, optional): 页码范围，如 (1, 10)。默认为 None，表示全部页面
    
    Returns:
        dict: 包含提取结果的字典
        {
            "success": bool,
            "text": str,           # 提取的文本内容
            "pages": int,          # 总页数
            "error": str           # 错误信息（如果失败）
        }
    
    Raises:
        FileNotFoundError: 当 PDF 文件不存在时
        ValueError: 当页码范围无效时
    """
    pass
```

### 4.3 代码风格规范

- 遵循 **PEP 8** 编码规范
- 使用 **类型注解** (Type Hints)
- 变量命名清晰，避免单字母变量（循环除外）
- 复杂逻辑添加行内注释

---

## 5. 隐身协议

**以上开发过程必须保持「影子模式」**。

### 5.1 代码痕迹
- 所有代码需表现为**个人开发痕迹**
- 提交信息、注释、文档均体现个人风格

### 5.2 严禁提及
- **严禁**在任何代码、注释、提交信息、文档中提及 AI 辅助信息
- **严禁**包含任何与 AI 生成相关的标识或元数据

### 5.3 合规示例

```python
# ✅ 正确的注释风格
# 使用余弦相似度计算文本匹配度
def calculate_similarity(text1: str, text2: str) -> float:
    pass

# ❌ 错误的注释风格
# AI 生成的代码：使用余弦相似度计算文本匹配度
def calculate_similarity(text1: str, text2: str) -> float:
    pass
```

```bash
# ✅ 正确的提交信息
git commit -m "feat: 实现论文与代码的相似度匹配算法"

# ❌ 错误的提交信息
git commit -m "feat: AI 生成的相似度匹配算法"
```

---

## 附录：开发检查清单

### 功能开发前
- [ ] 明确功能需求和输入输出
- [ ] 设计模块接口
- [ ] 编写 docstring 草稿

### 功能开发中
- [ ] 遵循目录结构规范
- [ ] 添加类型注解
- [ ] 编写单元测试

### 功能开发后
- [ ] 自测通过
- [ ] 更新文档
- [ ] 执行 Git Commit
- [ ] 检查提交信息合规性

---

*本文档为 LabFlow AI 项目开发规范，请严格遵守*
