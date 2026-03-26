# 任务文档：演示链路收尾优化

## 任务目标

围绕比赛演示阶段做低风险优化，不再引入新功能，只处理以下三类问题：

1. 结果加载反馈不够清晰，用户容易误以为页面卡死。
2. 导读页头部和正文卡片仍偏松散，视频观感不够收敛。
3. 演示链路缺少防呆提示，PDF 预览失败、代码路径错误、未配置模型时不够直观。

## 方案约束

- 不改动现有论文-代码对齐主链路的核心策略。
- 不再用“减少内容输出”换速度，阅读笔记保持完整生成路径。
- 所有优化都优先体现在 UI 状态提示与页面结构上，降低演示翻车概率。

## 实施范围

- [D:\Labflow\src\labflow\ui\app.py]
  - 工作区顶部增加更明确的运行阶段提示。
  - 阅读笔记增加阶段性进度说明与生成完成提示。
  - PDF 解析错误、代码路径错误、未配置模型时补充明确防呆文案。
- [D:\Labflow\src\labflow\ui\guide_page.py]
  - 导读页头部改为更紧凑的标题 + 状态说明 + 操作按钮。
  - 导读页只保留“论文概览 + 导读报告”两块主内容。
- [D:\Labflow\src\labflow\ui\styles.py]
  - 收紧导读页标题、说明和报告内容的间距与字号层级。
- [D:\Labflow\tests\unit\test_guide_page.py]
  - 增加导读页顶部状态文案测试。
- [D:\Labflow\tests\unit\test_ui_app.py]
  - 增加工作区运行状态文案映射测试。

## TODO

- [x] 为工作区代码对齐增加可读的阶段提示文案
- [x] 为阅读笔记保留完整生成路径，同时补充进度反馈
- [x] 为 PDF/代码路径/模型配置补充防呆提示
- [x] 收紧导读页头部和正文布局
- [x] 补齐相关单元测试并完成格式化、lint、pytest 校验

## 验证要求

- `python -m ruff check app.py src tests .agentdocs`
- `python -m ruff format --check app.py src tests .agentdocs`
- `python -m pytest`
