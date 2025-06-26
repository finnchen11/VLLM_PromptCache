# 🧠 vLLM System Prompt Cache —— 系统提示词缓存优化模块

> ⚡ 基于 vLLM 的系统提示词（System Prompt）缓存与复用机制，提升多轮对话场景下的推理性能。

---

## 📌 项目简介

本项目基于 [vLLM](https://github.com/vllm-project/vLLM) 进行扩展，实现了 **系统提示词永不释放 + 缓存复用机制**，特别适用于以下场景：

- 多用户共享相同系统提示并且系统提示词长度特别大，预填充阶段耗时多
- LUR机制当没有请求是，就会释放cacahe，新来的请求又会重新计算
- 高频请求中重复使用相同 system prompt
- 需要节省显存并提高吞吐量的部署环境

通过该模块，你可以：

✅ **避免重复分配 KV Cache**  
✅ **显著减少内存占用与推理延迟**  
✅ **无缝兼容 OpenAI API 接口**

---

## 🔍 功能亮点

| 功能 | 描述 |
|------|------|
| ✅ 系统提示词缓存 | 自动识别 `role: "system"` 消息，并缓存其 KV Cache |
| ✅ 缓存永不释放 | 标记为 `is_system_prompt` 的序列不会被释放 |
| ✅ 缓存块复制复用 | 支持在多个请求之间高效复用缓存块 |
| ✅ 兼容 OpenAI SDK | 可通过 `openai.ChatCompletion.create()` 直接调用 |
| ✅ 性能优化 | 减少内存分配开销，提升并发吞吐能力 |

---

## 🛠️ 安装与部署

### 1. 克隆项目

```bash
git clone https://github.com/finnchen11/vllm-system-prompt-cache.git
cd vllm-system-prompt-cache
```

### 2. 安装依赖

建议使用 Python 3.8+ 和 PyTorch 2.0+

```bash
pip install -e .
```

或者从原始 vLLM 安装后手动替换文件。

### 3. 启动服务

```bash
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model your_model_name
```

---

## 📦 使用方式

### 1. 使用 curl 调用

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "讲个笑话"}
        ]
    }'
```

### 2. 使用 OpenAI SDK

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"

completion = openai.ChatCompletion.create(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "讲个笑话"},
    ]
)
print(completion.choices[0].message.content)
```

---

## 🧩 核心修改说明

以下是项目中主要修改或新增的文件及功能：

| 文件 | 修改内容 |
|------|----------|
| `sequence.py` | 新增 `is_system_prompt` 字段用于标记系统提示词 |
| `worker/cache_engine.py` | 实现 `free()` 中判断是否为系统提示词，不释放缓存；添加 `copy()` 方法支持缓存块复用 |
| `engine/llm_engine.py` | 添加 `add_system_prompt()` 方法，实现系统提示词缓存池管理 |
| `entrypoints/openai/api_server.py` | 在 `/v1/chat/completions` 接口中提取 system 提示词并调用缓存机制 |

---

## 📊 性能对比（示意）

| 场景 | 原始 vLLM | 加入系统提示词缓存后 |
|------|------------|---------------------|
| 内存占用 | 高 | 显著降低 |
| 单次推理时间 | 高（首次缓存未命中） | 快速（后续命中缓存） |
| 并发吞吐量 | 一般 | 明显提升 |

---
