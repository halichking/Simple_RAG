from typing import TypedDict

import config_data as info
from file_history_store import get_history
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from vector_stores import VectorStoreService

class RagState(TypedDict, total=False):
    question: str
    session_id: str
    history: list[BaseMessage]
    documents: list[Document]
    context: str
    answer: str


class RagService(object):
    def __init__(self):
        embedding = DashScopeEmbeddings(
            model=info.embedding_model_name,
            dashscope_api_key=info.dashscope_api_key,
        )
        self.retriever = VectorStoreService(embedding).get_retriever()
        self.llm = ChatOllama(
            model=info.ollama_chat_model_name,
            base_url=info.ollama_base_url,
            temperature=0.2,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RagState)
        graph.add_node("load_history", self._load_history)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("save_history", self._save_history)

        graph.add_edge(START, "load_history")
        graph.add_edge("load_history", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "save_history")
        graph.add_edge("save_history", END)
        return graph.compile()

    def ask(self, question: str, session_id: str | None = None) -> str:
        result = self.graph.invoke(
            {
                "question": question,
                "session_id": session_id or info.default_session_id,
            }
        )
        return result["answer"]

    def _load_history(self, state: RagState) -> RagState:
        history = get_history(state["session_id"]).messages
        return {"history": history}

    def _retrieve(self, state: RagState) -> RagState:
        documents = self.retriever.invoke(state["question"])
        return {
            "documents": documents,
            "context": self._format_documents(documents),
        }

    def _generate(self, state: RagState) -> RagState:
        system_prompt = (
            "你是公司内部知识库助手。请优先根据参考资料回答问题；"
            "如果参考资料无法回答，就明确说明当前资料无法确定。\n\n"
            "参考资料：\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm

        response = chain.invoke({
            "context": state["context"],
            "history": state.get("history", []),
            "question": state["question"]
        })
        
        return {"answer": response.content}

    def _save_history(self, state: RagState) -> RagState:
        history_store = get_history(state["session_id"])
        history_store.add_messages(
            [
                HumanMessage(content=state["question"]),
                AIMessage(content=state["answer"]),
            ]
        )
        return {}

    def _format_documents(self, documents: list[Document]) -> str:
        if not documents:
            return "无相关参考资料"

        parts = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "未知来源")
            parts.append(
                f"资料{index}\n来源：{source}\n内容：{document.page_content}"
            )
        return "\n\n".join(parts)


if __name__ == "__main__":
    question = input("请输入问题：").strip()
    if question:
        print(RagService().ask(question))
