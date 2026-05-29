# RAG

一个本地知识库 RAG 项目：使用 LangGraph 编排问答流程，Chroma 保存本地向量库，Ollama 调用公司内网模型生成回答。

## 主要模块

- `src/config_data.py`：读取本地配置和环境变量
- `src/knowledge_base.py`：将 `docs` 文档写入向量库
- `src/vector_stores.py`：封装 Chroma 检索器
- `src/file_history_store.py`：保存本地会话历史
- `src/rag.py`：LangGraph RAG 问答流程

## 本地配置

敏感文件不会提交到仓库。请在项目根目录自行创建：

- `.env`：保存 `DASHSCOPE_API_KEY`
- `config_data.yml`：保存路径、模型名、切分参数等本地配置

`config_data.yml` 需要包含这些配置段：

```yaml
paths:
  md5_path: data/md5.txt
  docs_dir: docs
  data_dir: data
  vector_store_dir: data/vector_store
  chat_history_dir: data/chat_history

models:
  embedding_model: text-embedding-v4
  ollama_base_url: http://你的Ollama地址:11434
  ollama_chat_model: deepseek-r1:14b

vector_store:
  collection_name: company_knowledge

splitter:
  chunk_size: 1000
  chunk_overlap: 150
  max_split_char_number: 1000
  separators: ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""]

retriever:
  top_k: 4
  similarity_threshold: 4

session:
  default_session_id: local_user_001
```

## 使用

```powershell
.\.venv\Scripts\python.exe src\config_data.py
.\.venv\Scripts\python.exe src\knowledge_base.py
.\.venv\Scripts\python.exe src\rag.py
```
