# 部署本地向量化和重排模型
> PandaWiki 本地 BGE 向量模型 + 重排模型 部署完整指南（Windows）

---

# 一、目标说明

本指南用于在 **Windows 本地环境** 部署：

* ✅ BGE 向量模型（Embedding）
* ✅ BGE 重排模型（Reranker）
* ✅ 本地 FastAPI 服务
* ✅ 提供 API URL + Token 给 PandaWiki 使用

最终实现：

```
PandaWiki
    ↓
HTTP API (http://127.0.0.1:8000)
    ↓
FastAPI 服务 (bge_server.py)
    ↓
Embedding 模型 + Reranker 模型
```

---

# 二、所使用模型说明

本方案使用由
BAAI 发布的 BGE 系列模型，托管在
Hugging Face 平台。

推荐模型：

### 1️⃣ 向量模型（Embedding）

* bge-small-zh-v1.5
* 向量维度：384
* 适合本地 CPU

### 2️⃣ 重排模型（Reranker）

* bge-reranker-base
* 用于对检索结果进行语义重排

---

# 三、Python 版本要求

要求：

```
Python 3.10 或 3.11
```

⚠ 不建议使用 3.12（部分依赖可能兼容性问题）

---

# 四、项目目录结构

建议结构：

```
bge-service/
│
├── bge_server.py
├── requirements.txt
├── models/
│   ├── bge-small-zh-v1.5/
│   └── bge-reranker-base/
```

---

# 五、创建虚拟环境（Windows）

进入项目目录：

```bash
python -m venv bge_env
```

激活环境：

```bash
bge_env\Scripts\activate
```

成功后终端前会出现：

```
(bge_env)
```

---

# 六、安装依赖

安装：

```bash
pip install -r requirements.txt
```

安装完成后可验证：

```bash
pip list
```

应看到：

* torch==2.10.0+cpu
* transformers==5.2.0
* sentence-transformers==5.2.3
* fastapi==0.133.1
* uvicorn==0.41.0
* ......

---

# 七、下载模型

## 1️⃣ 下载向量模型
在当前虚拟环境下运行
```
hf download BAAI/bge-base-zh-v1.5
```

---
## 2️⃣ 下载重排模型
在当前虚拟环境下运行
```
hf download BAAI/bge-reranker-base
```

---

# 八、验证模型下载成功

进入模型目录（默认缓存目录：`C:\Users\你的用户名\.cache\huggingface\hub\`），应看到：

```
models--BAAI--bge-base-zh-v1.5
models--BAAI--bge-reranker-base
```

每个模型目录结构如下
```
model
    └── snapshots
            └── xxxxxxxxxxxxx
                    config.json
                    model.safetensors
                    tokenizer.json
```

---

## 验证向量模型可加载

在当前虚拟环境执行：

```
python
```

进入环境后执行：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
print("Embedding model loaded")

vec = model.encode(["测试一句话"])
print(len(vec[0]))
```

如果输出：

```bash
Embedding model loaded
768
```

说明：
✅ embedding 模型正常
✅ 输出维度 768
✅ 可以用于 PandaWiki

---

## 验证重排模型可加载

在当前虚拟环境执行：

```
python
```

进入环境后执行：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

model.eval()

inputs = tokenizer(
    [["什么是RAG？", "RAG是一种检索增强生成方法"]],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    scores = model(**inputs).logits

print(scores)
```

如果输出一个类似：

```
tensor([[X.XXXX]])
```

说明：

✅ reranker 正常
✅ forward 推理成功

---

# 九、bge_server.py 功能说明

该服务提供三个接口：

| 接口      | 功能     |
| ------- | ------ |
| /embed  | 生成向量   |
| /rerank | 文档重排   |
| /health | 服务健康检查 |

---

# 十、启动服务

在虚拟环境下运行：

```bash
uvicorn bge_server:app --host 0.0.0.0 --port 8000
```

若看到：

```
Uvicorn running on http://0.0.0.0:8000
```

说明服务启动成功。

---

# 十一、验证服务运行


浏览器访问：

```
http://127.0.0.1:8000/docs
```

可看到 API 文档界面。

---

# 十二、Embedding API 说明

### 请求

POST

```
http://127.0.0.1:8000/embed
```

请求体：

```json
{
  "texts": ["工业AI是什么"]
}
```

---

### 返回

```json
{
  "embeddings": [
    [0.0123, -0.2234, ...]
  ]
}
```

说明：

* 每个输入文本对应一个向量
* small 模型输出 384 维向量
* 返回为二维数组

---

# 十三、Rerank API 说明

### 请求

POST

```
http://127.0.0.1:8000/rerank
```

请求体：

```json
{
  "query": "工业AI是什么",
  "documents": [
    "AI用于工业质检",
    "今天天气很好"
  ]
}
```

---

### 返回

```json
{
  "scores": [0.92, 0.03]
}
```

说明：

* 分数越高相关性越强
* PandaWiki 会根据分数排序

---

# 十四、API Token 机制（可选）

默认无鉴权。

若需要简单鉴权：

在 bge_server.py 添加：

```python
API_TOKEN = "your_secret_token"
```

并在请求头中携带：

```
Authorization: Bearer your_secret_token
```

---

# 十五、PandaWiki 配置示例

---

## Embedding 配置

```
API URL: http://127.0.0.1:8000/embed
API Key: your_secret_token
```

---

## Reranker 配置

```
API URL: http://127.0.0.1:8000/rerank
API Key: your_secret_token
```

---

# 十六、常见问题

---


# 十七、完整流程总结

1. 创建虚拟环境
2. 安装 requirements.txt
3. 下载 embedding 模型
4. 下载 reranker 模型
5. 验证模型加载
6. 启动 FastAPI
7. 验证 /docs
8. 在 PandaWiki 中配置 API