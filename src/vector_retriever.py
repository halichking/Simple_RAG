import ConfigData as info
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

KNOWLEDGE_COLLECTION = info.knowledge_collection_name
HISTORY_COLLECTION = info.history_collection_name

MODEL_NAME = info.embedding_model_name
API_KEY = info.dashscope_api_key
PERSIST_DIR = info.persist_directory
RETRIEVER_TOP_K = info.retriever_top_k


def _extract_query_terms(question: str) -> list[str]:
    """
    从用户问题中提取适合做关键词重排的通用资料词。

    纯向量检索有时会把目录页或相邻配置页排在前面；这些词用于把真正包含
    用户关键表达的原文片段提前，提升 PDF/TXT 资料问答命中率。
    """
    domain_terms = [
        "使用说明", "操作步骤", "操作流程", "功能说明", "数据查询", "状态查询",
        "新增记录", "创建记录", "处理记录", "完成记录", "归档记录",
        "登录", "配置", "权限", "页面", "模块", "流程", "步骤", "查询",
        "新增", "创建", "编辑", "删除", "处理", "完成", "归档", "客户档案",
    ]
    terms = [term for term in domain_terms if term in question]
    if "操作" in question and "步骤" in question and "操作步骤" not in terms:
        terms.insert(0, "操作步骤")
    if "查询" in question and "状态" in question and "状态查询" not in terms:
        terms.insert(0, "状态查询")
    return terms

class VectorRetrieveService(object):
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=KNOWLEDGE_COLLECTION,
            embedding_function=DashScopeEmbeddings(model=MODEL_NAME, dashscope_api_key=API_KEY),
            persist_directory=PERSIST_DIR
        )
        self.history_store = Chroma(
            collection_name=HISTORY_COLLECTION,
            embedding_function=DashScopeEmbeddings(model=MODEL_NAME, dashscope_api_key=API_KEY),
            persist_directory=PERSIST_DIR
        )

    def get_retriever(self):
        """获取知识库的标准检索器（供 rag.py 传统节点调用）"""
        return self.vector_store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    def search_knowledge(self, question: str, k: int | None = None) -> list[Document]:
        """
        知识库混合检索：向量召回 + 关键词命中重排。

        第一阶段用 embedding 做语义召回；第二阶段扫描当前知识库切片，把包含用户问题
        关键表达的片段补进候选集并前置。这样能降低“资料里明明有相关流程，
        但语义检索先命中目录页/配置页”的概率。
        """
        top_k = k or RETRIEVER_TOP_K
        semantic_docs = self.vector_store.similarity_search(question, k=max(top_k * 3, 8))
        terms = _extract_query_terms(question)

        keyword_docs: list[Document] = []
        if terms:
            try:
                raw = self.vector_store._collection.get(include=["documents", "metadatas"])
                for content, metadata in zip(raw.get("documents", []), raw.get("metadatas", [])):
                    if not content:
                        continue
                    if any(term in content for term in terms):
                        keyword_docs.append(Document(page_content=content, metadata=metadata or {}))
            except Exception as e:
                print(f" [关键词检索异常，将仅使用向量检索]: {str(e)}")

        merged_docs = keyword_docs + semantic_docs
        seen_keys: set[tuple[str, str]] = set()
        scored_docs: list[tuple[int, int, Document]] = []
        for order, doc in enumerate(merged_docs):
            source = doc.metadata.get("source", "")
            content = doc.page_content or ""
            dedupe_key = (source, content[:200])
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            # 命中的关键词越多越靠前；较长组合词比单字词更有价值，因此按词长加权。
            score = sum(len(term) for term in terms if term in content)
            scored_docs.append((score, -order, doc))

        scored_docs.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [doc for _, _, doc in scored_docs[:top_k]]

    def knowledge_count(self) -> int:
        """返回知识库向量数量，用于健康检查和前端状态展示。"""
        return self.vector_store._collection.count()

    def history_count(self) -> int:
        """返回历史记忆向量数量，用于健康检查和前端状态展示。"""
        return self.history_store._collection.count()

    def delete_knowledge_base(self) -> dict[str, int]:
        """
        清空知识库向量集合，并删除 md5 入库记录。

        只清理 Chroma 中的 Learning_Knowledge 集合数据，不删除 docs 原始文件，也不删除
        Chat_History 会话记忆集合。
        """
        raw = self.vector_store._collection.get()
        ids = raw.get("ids", [])
        deleted_count = len(ids)
        if ids:
            self.vector_store._collection.delete(ids=ids)

        md5_deleted = 0
        md5_path = Path(info.md5_path)
        if md5_path.exists():
            md5_path.unlink()
            md5_deleted = 1

        return {"deleted_vectors": deleted_count, "deleted_md5_files": md5_deleted}

    def add_chat_history_round(self, question: str, answer: str, session_id: str):
        """将被剪枝的一问一答按 session_id 存入长期历史向量库。"""
        combined_text = f"历史用户问题：{question}\n历史AI回答：{answer}"
        metadata = {
            "session_id": session_id,
            "store_type": "chat_history"
        }

        self.history_store.add_texts(
            texts=[combined_text],
            metadatas=[metadata]
        )

    def search_chat_history(self, question: str, session_id: str, k: int = 2) -> str:
        """带安全隐私过滤的远期语义记忆召回"""
        search_kwargs = {
            "k": k,
            "filter": {"session_id": session_id}
        }

        try:
            docs = self.history_store.similarity_search(question, **search_kwargs)
            if not docs:
                return ""

            # 组装格式化字符串
            parts = [f"--- 相关远期对话历史片段 ---\n{doc.page_content}" for doc in docs]
            return "\n\n".join(parts)
        except Exception as e:
            print(f" [远期历史检索异常]: {str(e)}")
            return ""

    def delete_chat_history(self, session_id: str):
        """根据 session_id，将该用户在本地Chroma向量库里的长期记忆删除"""
        try:
            self.history_store._collection.delete(where={"session_id": session_id})
        except Exception as e:
            print(f"[Chroma 删除异常] 历史向量删除失败: {str(e)}")
