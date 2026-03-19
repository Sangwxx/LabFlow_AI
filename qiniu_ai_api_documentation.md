# 七牛 AI 大模型推理 API 文档

> 来源: [七牛开发者中心](https://developer.qiniu.com/aitokenapi)  
> 更新时间: 2025-12-29  
> 文档整理时间: 2026-03-19

---

## 目录

1. [Token API 实时推理接口](#一-token-api-实时推理接口)
2. [聊天补全接口](#二-聊天补全接口)
3. [全网搜索 API](#三-全网搜索-api)
4. [OCR 图片文档识别 API](#四-ocr-图片文档识别-api)
5. [MCP 接入服务](#五-mcp-接入服务)

---

## 一、Token API 实时推理接口

### 1.1 接入点

```
https://api.qnaigc.com/v1
```

### 1.2 前置条件

需要先 [获取 API KEY (API 密钥)](https://developer.qiniu.com/aitokenapi/12884/how-to-get-api-key)

### 1.3 支持接口列表

| 接口 | 说明 |
|------|------|
| `/v1/chat/completions` | 对话型推理接口（兼容 OpenAI 格式），支持图片文字识别、文件识别 |
| `/v1/models` | 列举所有可用模型 ID 及参数 |
| `/v1/messages` | 兼容 Anthropic API 格式 |

### 1.4 获取可用模型列表

```bash
curl https://api.qnaigc.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 1.5 对话接口调用示例

#### HTTP 调用

```bash
curl https://api.qnaigc.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己"}
    ]
  }'
```

#### Python OpenAI SDK 调用

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.qnaigc.com/v1"
)

# 非流式调用
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)
print(response.choices[0].message.content)

# 流式调用
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 1.6 图像生成

向 `/v1/chat/completions` 发送请求，设置 `modalities` 参数：

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash-image",
    modalities=["image", "text"],
    messages=[
        {"role": "user", "content": "生成一只可爱的猫咪图片"}
    ],
    image_config={
        "aspect_ratio": "1:1"  # 可选
    }
)
```

#### 支持的宽高比

| 比例 | 尺寸 |
|------|------|
| 1:1 | 1024×1024（默认） |
| 2:3 | 832×1248 |
| 3:2 | 1248×832 |
| 3:4 | 864×1184 |
| 4:3 | 1184×864 |
| 4:5 | 896×1152 |
| 5:4 | 1152×896 |
| 9:16 | 768×1344 |
| 16:9 | 1344×768 |
| 21:9 | 1536×672 |

### 1.7 文件识别推理

支持 PDF、DOCX、XLSX、PPTX 格式（内容受 64K tokens 限制）：

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请分析这个文档"},
                {
                    "type": "file",
                    "file": {
                        "type": "pdf",
                        "url": "https://example.com/document.pdf"
                    }
                }
            ]
        }
    ]
)
```

### 1.8 图片文字识别推理

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ]
)
```

**注意**: 图片文件大小不能超过 8MB，支持 JPG、JPEG、PNG、BMP、PDF 格式。

---

## 二、聊天补全接口

### 2.1 概述

聊天补全接口用于生成对话模型的响应，支持多种模型配置和生成控制参数，兼容 OpenAI 和 Anthropic 标准。

### 2.2 请求参数

#### 基本信息参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `model` | string | 是 | - | 模型名称，如 "gemini-2.5-flash" |
| `messages` | array | 是 | - | 对话消息数组 |
| `stream` | boolean | 否 | false | 是否启用流式响应 |

#### 消息格式

```json
{
  "role": "user",           // 可选: system, user, assistant, tool
  "content": "消息内容",     // 字符串或复杂内容对象
  "name": "参与者名称",      // 可选
  "tool_calls": [],         // 可选: 工具调用信息
  "tool_call_id": ""        // 可选: 工具调用ID（role为tool时）
}
```

#### 生成控制参数

| 参数名 | 类型 | 必填 | 默认值 | 范围 | 说明 |
|--------|------|------|--------|------|------|
| `max_tokens` | integer | 否 | 模型默认 | ≥1 | 生成的最大令牌数 |
| `temperature` | float | 否 | 1.0 | [0.0, 2.0] | 采样温度，越高越随机 |
| `top_p` | float | 否 | 1.0 | [0.0, 1.0] | 核采样参数 |
| `top_k` | integer | 否 | - | ≥1 | 仅从概率最高的k个令牌采样 |
| `presence_penalty` | float | 否 | 0.0 | [-2.0, 2.0] | 存在惩罚，鼓励新内容 |
| `frequency_penalty` | float | 否 | 0.0 | [-2.0, 2.0] | 频率惩罚 |
| `repetition_penalty` | float | 否 | 1.0 | [0.0, 2.0] | 重复惩罚 |

#### 思维链参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `thinking` | object | 否 | 思维链配置，如 `{"type": "enabled", "budget_tokens": 160}` |
| `reasoning_effort` | string | 否 | 推理强度: "low", "medium", "high" |

#### 工具调用参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `tools` | array | 否 | 可供模型调用的工具定义列表 |
| `tool_choice` | string/object | 否 | 工具调用策略: "none", "auto", "required" 或指定具体工具 |

#### 多模态支持

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `image_config` | object | 否 | 图像输入配置 |
| `response_format` | object | 否 | 响应格式控制，如 `{"type": "json_object"}` |

### 2.3 参数最佳实践

| 场景 | 推荐设置 |
|------|----------|
| 创造性任务 | temperature: 0.7-0.9 |
| 事实性回答 | temperature: 0.1-0.3 |
| 代码生成 | temperature: 0.1-0.3 |
| 减少重复 | repetition_penalty: 1.1-1.2 |
| 鼓励多样性 | presence_penalty: 0.1-0.5 |
| 对话场景 | max_tokens: 500-1000 |

### 2.4 错误处理

| 错误码 | 说明 |
|--------|------|
| 400 | 参数验证失败 |
| 401 | 认证失败 |
| 429 | 速率限制 |
| 500 | 服务器内部错误 |

---

## 三、全网搜索 API

### 3.1 简介

提供高质量的全网搜索服务（基于百度 Search API），支持多种搜索类型和过滤条件，返回结构化的搜索结果。

### 3.2 API 接入点

```
https://ai.qiniuapi.com/v1/search
```

### 3.3 请求方式

- **方法**: POST
- **Content-Type**: `application/json`

### 3.4 请求参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `query` | string | 是 | 搜索关键词或查询语句 |
| `max_results` | int | 否 | 返回结果数量。网页默认20/最大50；视频最大10（默认5）；图片最大30（默认15） |
| `search_type` | string | 否 | 搜索类型: "web"(默认), "video", "image" |
| `time_filter` | string | 否 | 时间过滤: "week", "month", "year", "semiyear" |
| `site_filter` | array | 否 | 站点过滤，最多20个站点 |

### 3.5 搜索类型说明

| 类型 | 说明 |
|------|------|
| `web` | 网页搜索（默认） |
| `video` | 视频搜索 |
| `image` | 图片搜索 |

### 3.6 请求示例

```bash
curl -X POST "https://ai.qiniuapi.com/v1/search" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "人工智能发展趋势",
    "max_results": 20,
    "search_type": "web",
    "time_filter": "month"
  }'
```

### 3.7 响应结构

#### 顶层响应

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `success` | boolean | 请求是否成功 |
| `message` | string | 错误信息（失败时返回） |
| `data` | object | 搜索结果数据 |

#### 搜索结果数据

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `query` | string | 搜索查询词 |
| `results` | array | 搜索结果列表 |
| `total` | int | 搜索结果总数 |
| `request_id` | string | 本次请求的唯一标识 |

#### 搜索结果项

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | int | 结果项ID |
| `title` | string | 页面标题 |
| `url` | string | 页面链接 |
| `content` | string | 页面摘要内容 |
| `date` | string | 发布时间 |
| `source` | string | 来源网站名称 |
| `score` | float | 相关性评分 |
| `type` | string | 结果类型（web/news等） |
| `icon` | string | 网站图标链接 |
| `authority_score` | float | 权威性评分 |
| `image` | object | 图片信息（图片搜索时），包含url/height/width |
| `video` | object | 视频信息（视频搜索时），包含url/height/width/size/duration/hover_pic |

---

## 四、OCR 图片文档识别 API

### 4.1 简介

支持对图片和 PDF 文档进行高精度文字识别（OCR），具备超低延迟响应。

### 4.2 功能特性

- 支持多种输入格式：PNG、JPG、PDF 等
- 高精度文字提取
- 超低延迟响应
- 标准 RESTful API

### 4.3 API 接入点

```
https://api.qnaigc.com/v1
```

### 4.4 请求示例

```bash
curl -X POST "https://api.qnaigc.com/v1/ocr" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ocr",
    "url": "https://example.com/document.pdf"
  }'
```

### 4.5 请求参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `model` | string | 是 | 固定为 "ocr" |
| `url` | string | 是 | 需识别图片或 PDF 的公网链接 |

### 4.6 响应字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `text` | string | 识别出的全部文本内容 |
| `id` | string | 本次调用的ID |

---

## 五、MCP 接入服务

### 5.1 简介

MCP（模型上下文协议）接入服务是为各类大模型推理服务提供统一、安全、标准化接入与编排的中间层。

### 5.2 适用场景

- 一步调用 LLM 推理大模型和多个流行 MCP 工具
- 灵活编排、聚合多种工具和模型
- 集中安全托管多种 MCP 的敏感密钥
- 解决本地终端系统多样难以配置 MCP 服务的问题

### 5.3 接入方式

#### 1. Agent 协议接入（兼容 OpenAI）

```
https://api.qnaigc.com/v1/agent/instance/${mcp-id}
```

**多服务聚合**:
```
https://api.qnaigc.com/v1/agent/group/${mcp-id-1},${mcp-id-2},${mcp-id-3}
```

#### 2. 标准 MCP 协议接入

| 协议 | 接入地址 |
|------|----------|
| SSE | `https://api.qnaigc.com/v1/mcp/sse/${mcp-id}` |
| HTTP-Streamable | `https://api.qnaigc.com/v1/mcp/http-streamable/${mcp-id}` |

### 5.4 核心优势

- **协议标准化**: 支持 OPENAI、SSE、HTTP-Streamable 等主流协议
- **安全托管**: API Key、密钥等敏感信息云端托管
- **灵活聚合**: 支持多服务聚合调用
- **统一管理**: 可视化控制台统一管理

### 5.5 主要功能

- MCP 服务接入与协议转换
- 安全密钥托管与权限管理
- 多服务聚合与编排
- 可视化服务管理

### 5.6 使用流程

1. 登录七牛云 AI 控制台，进入 MCP 服务管理页面
2. 添加或管理 MCP 服务，获取专属的 MCP-ID
3. 根据需求选择接入协议（Agent/SSE/HTTP-Streamable）
4. 使用生成的接入地址进行调用

### 5.7 Agent 协议调用示例

```bash
curl https://api.qnaigc.com/v1/agent/instance/${mcp-id}/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

### 5.8 注意事项

- Agent 协议下无需本地安装任何 MCP 组件
- MCP 接入服务本身不直接提供大模型能力，而是作为能力编排中间层
- API Key 请妥善保管，避免泄露

---

## 附录：相关链接

- [七牛 AI 大模型推理产品介绍](https://www.qiniu.com/ai)
- [模型广场](https://www.qiniu.com/ai/models)
- [获取 API KEY 文档](https://developer.qiniu.com/aitokenapi/12884/how-to-get-api-key)
- [对象存储 SDK 中心](https://developer.qiniu.com/sdk#official-sdk)
- [对象存储产品文档](https://developer.qiniu.com/kodo/1312/upload)

---

*本文档基于七牛开发者中心官方文档整理*
