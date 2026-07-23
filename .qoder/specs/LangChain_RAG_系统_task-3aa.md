# 私人食谱与饮食规划师 — RAG 系统实施方案

## Context

构建一个基于 LangChain 的 RAG 系统，主题为 **私人食谱与饮食规划师**。系统支持离线文档处理（预置食谱知识库 + 用户上传）、在线混合检索与流式生成，具备基于 Embedding 的短期会话记忆能力，并支持 **用户反馈回流知识库**（问答评价 → 管理员审核 → 优质内容入库）。前端包含聊天界面和反馈管理页面。

**技术栈**: Qwen-3.7-Plus (DashScope API) + DashScope Embedding + Chroma + FastAPI + 原生前端

---

## 项目目录结构

```
e:\Agent-project1\
├── .env                          # DASHSCOPE_API_KEY 等环境变量
├── config.yaml                   # 应用配置
├── requirements.txt
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI 入口, lifespan, 静态文件
│   ├── config.py                 # pydantic-settings 配置
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router_chat.py        # POST /api/chat (SSE 流式)
│   │   ├── router_documents.py   # 文档上传/列表/删除
│   │   ├── router_feedback.py    # 反馈提交/审核/管理 API
│   │   └── schemas.py            # Pydantic 模型
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── offline_pipeline.py   # 离线数据处理编排
│   │   ├── document_parser.py    # Markdown/TXT 解析
│   │   ├── text_splitter.py      # 文本分块
│   │   ├── embedding.py          # DashScope Embedding 封装
│   │   ├── vector_store.py       # Chroma 向量库
│   │   ├── bm25_retriever.py     # BM25 关键词检索
│   │   ├── hybrid_retriever.py   # 混合检索 (向量+BM25+RRF)
│   │   ├── reranker.py           # Cross-encoder 重排序
│   │   ├── memory.py             # 短期记忆 (embedding-based)
│   │   ├── feedback_store.py     # 反馈回流存储 (含审核状态管理)
│   │   ├── prompt_builder.py     # 饮食领域 Prompt 模板
│   │   └── rag_chain.py          # RAG 主链路编排
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── data/
│   ├── knowledge_base/           # 预置食谱知识库 (md/txt)
│   ├── raw/                      # 用户上传的原始文档
│   ├── feedback_qa/              # 用户反馈的Q&A存档(JSON)
│   └── chroma_db/                # Chroma 持久化目录
│
├── frontend/
│   ├── index.html                # 聊天界面 (一问一答)
│   ├── admin.html                # 反馈管理页面 (审核/查看/删除)
│   ├── style.css
│   └── app.js                    # SSE 流式接收 + 反馈按钮
│
└── scripts/
    ├── ingest.py                 # 离线导入脚本 (CLI)
    └── clear_db.py               # 清空向量库
```

---

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│           FRONTEND                                        │
│  ┌──────────────────┐   ┌────────────────────────────┐  │
│  │ index.html       │   │ admin.html                 │  │
│  │ 聊天界面          │   │ 反馈管理 (审核/查看/删除)    │  │
│  │ 一问一答 + 👍/👎  │   │ Tab切换 + 批量操作          │  │
│  └──────────────────┘   └────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│                  FASTAPI BACKEND                          │
│                                                          │
│  ┌──────────── RAG CHAIN ────────────────────────────┐   │
│  │  Query → 记忆检索 → Query改写 → 混合检索           │   │
│  │  → 重排序 → Prompt组装 → Qwen流式生成 → 存入记忆   │   │
│  └────────────────────────────────────────────────────┘   │
│                                                          │
│  ┌── 反馈回流 (含审核) ──────────────────────────────┐   │
│  │ 用户 👍 → pending → 管理员审核                      │   │
│  │ → approved: Embedding → 入向量库 → 可被检索         │   │
│  │ → rejected: 仅保留记录，不入向量库                   │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌── 离线 Pipeline ──────────────────────────────────┐   │
│  │ 预置知识库 + 用户上传 → 解析 → 分块 → Embedding     │   │
│  │ → Chroma 入库 + BM25 索引构建                      │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 核心数据流

### 离线流程
```
预置食谱(md/txt) + 用户上传文档
    → Document Parser → Text Splitter (512 chars, 64 overlap)
    → DashScope Embedding (text-embedding-v3)
    → Chroma 入库 (documents 集合) + BM25 索引构建
```

### 在线流程
```
用户提问 → 短期记忆检索 → Query 改写
    → 混合检索 (documents + feedback_qa + BM25 → RRF 融合)
    → Cross-Encoder 重排序 → top-5
    → Prompt 组装 → Qwen-3.7-Plus 流式生成 → SSE 推送
    → 本轮 (user, ai) 存入短期记忆
```

### 反馈回流流程（含审核）
```
AI 回答完成 → 前端显示 👍/👎 按钮
    → 用户点击 👍
    → 后端将 Q&A 对存入 feedback_qa，状态为 pending (待审核)
    → 管理员在 admin.html 查看待审核列表
    → 管理员点击「通过」:
        → status → approved
        → Embedding 向量化 → 入 Chroma feedback_qa + BM25 索引
        → 后续检索时可被召回
    → 管理员点击「拒绝」:
        → status → rejected，不入向量库
    → 管理员可「删除」已审核记录（从向量库移除）
```

---

## Task 1: 项目骨架与配置管理

**文件**: `.env`, `config.yaml`, `requirements.txt`, `app/config.py`, `app/main.py`

**config.yaml**:
```yaml
llm:
  model: "qwen-plus"
  temperature: 0.7
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
embedding:
  model: "text-embedding-v3"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
chunking:
  chunk_size: 512
  chunk_overlap: 64
retrieval:
  vector_top_k: 20
  bm25_top_k: 20
  rerank_top_k: 5
memory:
  turn_limit: 10
chroma:
  persist_dir: "./data/chroma_db"
```

**requirements.txt**:
```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-chroma>=0.2.0
dashscope>=1.20.0
chromadb>=0.5.0
rank-bm25>=0.2.2
jieba>=0.42.1
sentence-transformers>=3.0.0
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
pydantic-settings>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

---

## Task 2: 离线数据处理 Pipeline

**文件**: `app/core/document_parser.py`, `app/core/text_splitter.py`, `app/core/embedding.py`, `app/core/vector_store.py`, `app/core/offline_pipeline.py`

- **解析**: Markdown → UnstructuredMarkdownLoader, TXT → TextLoader
- **分块**: RecursiveCharacterTextSplitter(512, 64)
- **Embedding**: DashScope text-embedding-v3，batch 100 chunks
- **入库**: Chroma upsert，集合名 `documents`
- **BM25 索引**: 同步写入 `data/bm25_index.json`
- **预置知识库**: `data/knowledge_base/` 目录，ingest.py 自动扫描导入

---

## Task 3: 混合检索模块

**文件**: `app/core/bm25_retriever.py`, `app/core/hybrid_retriever.py`

- **BM25**: rank_bm25.BM25Okapi + jieba 中文分词，启动时加载到内存
- **RRF 融合**: score(d) = Σ 1/(60 + rank_i(d))，两路各 top-20 → 融合去重 top-15

---

## Task 4: 重排序模块

**文件**: `app/core/reranker.py`

- CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")，CPU 可运行
- 对 (query, doc) pairs 打分，返回 top-5

---

## Task 5: 短期记忆模块 (Embedding-based)

**文件**: `app/core/memory.py`

- Chroma 集合: `conversation_memory`
- **存储**: 每轮 user/ai 消息分别 embedding 存入，metadata 含 session_id, turn_number, role
- **检索**: 新 query 用 embedding 检索 top-3 相关历史
- **注入**: 检索到的历史作为上下文注入 prompt
- **淘汰**: 超过 turn_limit(10轮) 删除最旧记录

---

## Task 6: 反馈回流模块（含审核）

**文件**: `app/core/feedback_store.py`, `app/api/router_feedback.py`

### 反馈状态机
```
用户点 👍 → pending (待审核)
                ├─→ approved (通过) → Embedding → 入向量库 + BM25
                └─→ rejected (拒绝) → 仅保留记录
            approved → deleted (删除) → 从向量库移除
```

### 数据存储
- Chroma 集合: `feedback_qa`
- **只有 status=approved 的记录才参与检索**
- **去重**: 入库前检查 cosine similarity > 0.95 避免重复

### API 端点

| Method | Path | 描述 |
|--------|------|------|
| POST | `/api/feedback` | 提交评价 (→ pending) |
| GET | `/api/feedback/list` | 反馈列表 (status 筛选/分页) |
| PUT | `/api/feedback/{id}/approve` | 审核通过 → 入向量库 |
| PUT | `/api/feedback/{id}/reject` | 审核拒绝 |
| DELETE | `/api/feedback/{id}` | 删除 (从向量库移除) |
| GET | `/api/feedback/stats` | 统计 (待审核/已通过/已拒绝) |

---

## Task 7: RAG Chain + Prompt + 其他 API

**文件**: `app/core/rag_chain.py`, `app/core/prompt_builder.py`, `app/api/router_chat.py`, `app/api/router_documents.py`

**System Prompt**: 饮食规划师角色，含记忆上下文 + 参考资料 + 用户问题

**其他 API**:

| Method | Path | 描述 |
|--------|------|------|
| POST | `/api/chat` | 聊天 (SSE 流式) |
| POST | `/api/documents/upload` | 上传食谱文档 |
| GET | `/api/documents` | 列出已导入文档 |
| DELETE | `/api/documents/{doc_id}` | 删除文档 |
| POST | `/api/memory/clear` | 清空会话记忆 |

---

## Task 8: 前端页面

### 8.1 聊天界面 (index.html)
- 原生 HTML/CSS/JS + marked.js (CDN)
- fetch + ReadableStream 逐 token 接收
- 布局: 聊天消息气泡 + 参考来源 + 底部输入栏(含上传按钮)
- **反馈按钮**: 每条 AI 回答下方 👍/👎，点 👍 后显示「已提交审核」
- 响应式设计

### 8.2 反馈管理页面 (admin.html)
- 独立页面，通过导航栏切换
- **Tab 切换**: 待审核 / 已通过 / 已拒绝
- **每条反馈卡片**: Q&A 内容、提交时间、操作按钮（通过/拒绝/删除）
- **统计面板**: 顶部显示各状态数量
- **批量操作**: 支持全选通过/拒绝

---

## 验证方案

1. **离线 Pipeline**: 放入预置食谱 → 运行 ingest → 验证 Chroma + BM25 索引
2. **混合检索**: 用"低卡早餐"测试 → 验证融合排序
3. **记忆模块**: 连续对话 → 验证上下文关联
4. **流式输出**: 前端提问 → 验证逐字显示
5. **反馈回流**: 提问→回答→点 👍 → 验证 pending 记录 → 管理页面审核通过 → 验证入向量库 → 再次提问相似问题 → 验证召回反馈内容
6. **端到端**: 上传食谱 → 提问 → 引用正确段落 → 评价 → 管理页面审核 → 验证知识库增长
