# 七牛全网搜索 API 文档

> 来源: [七牛开发者中心](https://developer.qiniu.com/aitokenapi/13192/web-search-api)  
> 更新时间: 2025-12-23 17:11:53  
> 文档整理时间: 2026-03-19

---

## 简介

本接口提供高质量的全网搜索服务（基于百度 Search API），支持多种搜索类型和过滤条件，返回结构化的搜索结果，适用于智能问答、内容聚合、信息检索等多种应用场景。

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 多种搜索类型 | 支持网页搜索、图片搜索、视频搜索等多种搜索模式 |
| 灵活过滤 | 支持时间过滤、站点过滤等多维度筛选条件 |
| 结构化结果 | 返回标准化的搜索结果，包含标题、链接、摘要、来源等完整信息 |
| 高可用性 | 基于成熟的搜索引擎技术，提供稳定可靠的搜索服务 |
| 易于集成 | 标准 RESTful API，便于与各类系统对接 |

---

## 前置条件

使用本 API 前，需要先 [获取 API KEY (API 密钥)](https://developer.qiniu.com/aitokenapi/12884/how-to-get-api-key)。

---

## API 接入点

```
https://ai.qiniuapi.com/v1/search
```

---

## 请求方式

- **方法**: POST
- **Content-Type**: `application/json`
- **认证方式**: 通过 API Key 进行认证（具体认证方式参考七牛 API 文档）

---

## 请求参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `query` | string | 是 | 搜索关键词或查询语句 |
| `max_results` | int | 否 | 返回结果数量。网页搜索默认 20，最大 50；视频搜索最大 10（默认 5）；图片搜索最大 30（默认 15） |
| `search_type` | string | 否 | 搜索类型，默认 `"web"`（网页搜索） |
| `time_filter` | string | 否 | 时间过滤，可选值：`week`（一周内）、`month`（一月内）、`year`（一年内）、`semiyear`（半年内） |
| `site_filter` | array | 否 | 站点过滤，指定搜索特定网站的内容（最多 20 个） |

### 搜索类型说明

| 搜索类型 | 说明 |
|----------|------|
| `web` | 网页搜索（默认） |
| `video` | 视频搜索 |
| `image` | 图片搜索 |

---

## 请求示例

### cURL 示例

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

### Python 示例

```python
import requests

url = "https://ai.qiniuapi.com/v1/search"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
data = {
    "query": "人工智能发展趋势",
    "max_results": 20,
    "search_type": "web",
    "time_filter": "month"
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result)
```

---

## 响应结构

### 顶层响应

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `success` | boolean | 请求是否成功 |
| `message` | string | 错误信息（仅在失败时返回） |
| `data` | object | 搜索结果数据 |

### 搜索结果数据 (data)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `query` | string | 搜索查询词 |
| `results` | array | 搜索结果列表 |
| `total` | int | 搜索结果总数 |
| `request_id` | string | 本次请求的唯一标识 |

### 搜索结果项 (results 数组元素)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | int | 结果项 ID |
| `title` | string | 页面标题 |
| `url` | string | 页面链接 |
| `content` | string | 页面摘要内容 |
| `date` | string | 发布时间 |
| `source` | string | 来源网站名称 |
| `score` | float | 相关性评分 |
| `type` | string | 结果类型（`web`/`news` 等） |
| `icon` | string | 网站图标链接 |
| `authority_score` | float | 权威性评分 |
| `image` | object | 图片信息（图片搜索时），包含 `url`、`height`、`width` |
| `video` | object | 视频信息（视频搜索时），包含 `url`、`height`、`width`、`size`（字节）、`duration`（秒）、`hover_pic`（封面 URL）。注意：当 `url` 为空时，上级的 `url` 字段为视频平台链接 |

---

## 响应示例

```json
{
  "success": true,
  "data": {
    "query": "人工智能发展趋势",
    "total": 1000,
    "request_id": "req_abc123def456",
    "results": [
      {
        "id": 1,
        "title": "2025年人工智能发展趋势预测",
        "url": "https://example.com/article/1",
        "content": "人工智能技术正在快速发展，预计2025年将出现重大突破...",
        "date": "2025-12-20",
        "source": "科技日报",
        "score": 0.95,
        "type": "web",
        "icon": "https://example.com/favicon.ico",
        "authority_score": 0.88
      },
      {
        "id": 2,
        "title": "AI技术在各行业的应用现状",
        "url": "https://example.com/article/2",
        "content": "人工智能已广泛应用于医疗、教育、金融等多个领域...",
        "date": "2025-12-18",
        "source": "新华网",
        "score": 0.92,
        "type": "news",
        "icon": "https://example.com/favicon.ico",
        "authority_score": 0.90
      }
    ]
  }
}
```

### 图片搜索结果示例

```json
{
  "success": true,
  "data": {
    "query": "猫",
    "total": 500,
    "request_id": "req_xyz789abc",
    "results": [
      {
        "id": 1,
        "title": "可爱的猫咪图片",
        "url": "https://example.com/image/1",
        "content": "",
        "source": "图片网站",
        "score": 0.98,
        "type": "image",
        "image": {
          "url": "https://example.com/cat.jpg",
          "height": 800,
          "width": 1200
        }
      }
    ]
  }
}
```

### 视频搜索结果示例

```json
{
  "success": true,
  "data": {
    "query": "编程教程",
    "total": 200,
    "request_id": "req_video123",
    "results": [
      {
        "id": 1,
        "title": "Python入门教程",
        "url": "https://video-platform.com/watch?v=abc123",
        "content": "零基础学习Python编程",
        "source": "视频平台",
        "score": 0.96,
        "type": "video",
        "video": {
          "url": "",
          "height": 1080,
          "width": 1920,
          "size": 52428800,
          "duration": 1800,
          "hover_pic": "https://example.com/thumbnail.jpg"
        }
      }
    ]
  }
}
```

---

## 错误响应

当请求失败时，返回结构如下：

```json
{
  "success": false,
  "message": "错误描述信息"
}
```

### 常见错误码

| HTTP 状态码 | 说明 |
|-------------|------|
| 400 | 请求参数错误 |
| 401 | 认证失败，API Key 无效或过期 |
| 429 | 请求频率超限 |
| 500 | 服务器内部错误 |

---

## 使用限制

| 限制项 | 说明 |
|--------|------|
| 站点过滤数量 | 最多 20 个站点 |
| 网页搜索最大结果数 | 50 |
| 视频搜索最大结果数 | 10 |
| 图片搜索最大结果数 | 30 |

---

## 最佳实践

1. **合理设置结果数量**: 根据实际需求设置 `max_results`，避免获取过多无用数据
2. **使用时间过滤**: 对于时效性要求高的查询，使用 `time_filter` 获取最新内容
3. **站点过滤**: 如需特定来源的内容，使用 `site_filter` 提高结果相关性
4. **错误处理**: 始终检查 `success` 字段，并做好错误重试机制
5. **缓存结果**: 对于相同查询，适当缓存结果以减少 API 调用

---

## 相关链接

- [获取 API KEY 文档](https://developer.qiniu.com/aitokenapi/12884/how-to-get-api-key)
- [七牛 AI 大模型推理文档](https://developer.qiniu.com/aitokenapi)

---

*本文档基于七牛开发者中心官方文档整理*
