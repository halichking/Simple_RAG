# Simple RAG Assistant

一个本地运行的轻量 RAG 助手示例，支持资料问答、临时表格分析、流式输出和分层记忆管理。项目默认面向个人或小团队本地使用，不包含登录、权限、多租户等生产系统能力。

## 功能特性

- 本地网页聊天界面：标准库 `http.server` 实现，无需额外前端构建流程。
- 资料知识库问答：支持上传或同步 PDF/TXT，并写入 Chroma 向量库。
- 临时表格分析：聊天框左侧 `+` 上传 Excel/CSV，文件只绑定当前消息分析，不会自动混入长期知识库。
- 流式回答：`/chat-stream` 使用 NDJSON 分块返回，前端显示“思考中”等待态。
- 模型意图路由：通过结构化输出判断问题走资料问答、表格分析或普通聊天。
- 分层记忆：
  - 最近几轮消息作为短期窗口；
  - 旧消息滚动压缩进 `summary`；
  - 重要历史问答写入 Chroma 远期记忆；
  - 用户稳定信息抽取到 `profile`。
- 清理能力：
  - 清空当前会话历史、摘要、画像、远期记忆和临时文件；
  - 清空知识库向量和入库 MD5 记录。

## 项目结构

```text
.
├── config_data.example.yml   # 配置模板，复制为 config_data.yml 后使用
├── src/
│   ├── web_app.py            # 本地网页服务和 HTTP API
│   ├── rag.py                # RAG 核心服务、路由、流式问答
│   ├── memory_manager.py     # 分层记忆管理
│   ├── knowledge_base.py     # PDF/TXT 入库
│   ├── vector_retriever.py   # Chroma 检索、知识库/历史向量管理
│   ├── report_service.py     # Excel/CSV 读取和 Markdown 分析摘要
│   ├── session_file_service.py # 当前会话临时分析文件管理
│   ├── mongo_checkpointer.py # LangGraph MongoDB Checkpointer
│   ├── SchemaCollection.py   # Pydantic 结构化输出模型
│   ├── RagState.py           # LangGraph 状态定义
│   └── ConfigData.py         # 配置读取
└── README.md
```

## 环境要求

- Python 3.11 或更新版本
- MongoDB，本地或 Docker 均可
- 可访问的兼容 OpenAI Chat Completions 的模型服务
- DashScope Embedding API Key，用于向量化资料和历史记忆

当前代码使用的主要 Python 包包括：

- `langchain`
- `langchain-openai`
- `langgraph`
- `langgraph-checkpoint-mongodb`
- `langchain-chroma`
- `langchain-community`
- `pymongo`
- `python-dotenv`
- `pyyaml`
- `pydantic`
- `xlrd`，仅在读取 `.xls` 时需要

## 快速开始

1. 创建并激活虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 安装依赖。

如果你已经有自己的依赖管理文件，可以按自己的方式安装。否则可先按代码导入的包安装：

```powershell
pip install langchain langchain-openai langgraph langgraph-checkpoint-mongodb langchain-chroma langchain-community pymongo python-dotenv pyyaml pydantic xlrd
```

3. 准备配置文件。

```powershell
Copy-Item config_data.example.yml config_data.yml
```

按需修改 `config_data.yml` 中的模型、MongoDB、向量库路径等配置。

4. 准备环境变量。

在项目根目录创建 `.env`：

```env
ALIYUN_API_KEY=你的模型服务或 DashScope Key
DASHSCOPE_API_KEY=你的 DashScope Embedding Key
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

如果聊天模型和 Embedding 共用一个 Key，只设置 `ALIYUN_API_KEY` 也可以。

5. 启动 MongoDB。

Docker 示例：

```powershell
docker run -d --name local-rag-mongo -p 27017:27017 mongo:7
```

6. 启动网页服务。

```powershell
python src\web_app.py
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

## 使用方式

### 资料问答

点击页面右侧“上传资料到知识库”，上传 PDF/TXT。上传成功后即可提问文档流程、功能说明、字段含义等内容。

也可以把资料放入本地 `docs/` 目录，然后点击“同步 docs 知识库”。注意：`docs/` 默认被 `.gitignore` 忽略，避免误把本地资料提交到公开仓库。

### 表格分析

点击输入框左侧 `+`，选择 `.xlsx`、`.xls` 或 `.csv`，然后在同一条消息里输入问题并发送。

临时表格文件只用于当前会话分析，不会进入长期知识库。系统会先用程序读取、统计、生成 Markdown 摘要，再让模型基于统计结果做自然语言解读。

### 记忆机制

当前会话状态由 MongoDB Checkpointer 持久化。流式回答前会调用 `MemoryManager` 做分层记忆管理：

```text
短期窗口：保留最近几轮原文消息
滚动摘要：旧消息压缩进 summary
远期记忆：被剪枝的一问一答写入 Chroma
用户画像：从明确表达中抽取 profile
```

这样可以避免历史消息无限进入模型上下文，同时保留后续回答仍可能需要的关键信息。

### 清理数据

页面提供两个危险操作：

- 清空当前会话历史：清除当前 `session_id` 的消息、摘要、画像、远期记忆和临时文件。
- 清空知识库：清空知识库向量和 MD5 入库记录，不删除本地 `docs/` 原始文件。
