"""
基于 LangGraph + Docker MongoDB 托管的 RAG 核心服务
包含记忆功能
"""
import sys
import os
from pathlib import Path

from typing import Literal

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph, END

# 将项目根目录和 src 目录加入模块搜索路径，保证直接运行、LangGraph CLI 和网页服务都能找到本地模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
for module_path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import ConfigData as info
from RagState import RagState
from SchemaCollection import UserProfile
from SchemaCollection import RouteDecision
from memory_manager import MemoryManager
from vector_retriever import VectorRetrieveService
from mongo_checkpointer import get_mongodb_checkpointer
from report_service import ReportService
from session_file_service import SessionFileService

# LangSmith 追踪只读取 .env 中的配置，避免在源码中硬编码任何密钥。
if info.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = info.langsmith_api_key
if info.langsmith_project:
    os.environ["LANGCHAIN_PROJECT"] = info.langsmith_project

CHAT_MODEL_NAME = info.chat_model_name
API_KEY = info.chat_model_api_key
BASE_URL = info.chat_model_base_url


class RagService(object):
    def __init__(self):
        self.vector_service = VectorRetrieveService()
        self.report_service = ReportService()
        self.session_file_service = SessionFileService()
        self.llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=0.2,
            streaming=True
        )
        self.route_parser = PydanticOutputParser(pydantic_object=RouteDecision)
        # 最近 6 条消息约等于 3 轮问答；超过 12 条时把更早历史滚入摘要和远期向量记忆。
        self.memory_manage = MemoryManager(llm=self.llm, threshold=12, keep_recent=6)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = (
            StateGraph(RagState)
            .add_node("retrieve", self._retrieve)
            .add_node("generate", self._generate)
            .add_node("report", self._generate_report)
            .add_node("manage_memory", self.memory_manage.manage_pyramid_memory)
            .add_edge(START, "manage_memory")
            .add_conditional_edges(
                "manage_memory",
                self._route_intent,
                {
                    "knowledge_qa": "retrieve",
                    "data_report": "report",
                    "pure_chat": "generate"
                }
            )
            .add_edge("retrieve", "generate")
            .add_edge("report", END)
            .add_edge("generate", END)
        )
        return builder

    def _route_intent(self, state: RagState) -> Literal["knowledge_qa", "data_report", "pure_chat"]:
        """使用附件硬规则和模型结构化判断进行路由。"""
        question = self._resolve_referential_question(state)
        if state.get("analysis_file_names"):
            return "data_report"
        return self._decide_route_by_llm(question)

    def _decide_route_by_llm(self, question: str) -> Literal["knowledge_qa", "data_report", "pure_chat"]:
        """调用大模型按 RouteDecision 结构化输出进行意图路由。"""
        route_prompt = (
            "你是本地知识库助手的意图路由裁判。请只判断用户当前问题应该流向哪个节点，不要回答问题本身。\n\n"
            "【可选路由】\n"
            "1. knowledge_qa：用户在询问本地知识库、PDF/TXT 文档、产品手册、制度资料、使用说明、功能概念、操作步骤或字段含义。\n"
            "2. data_report：用户明确要求分析 Excel/CSV/临时附件/表格数据，或要求生成运营报表、统计数量、状态分布、完成率、超时数量、Top 排名、异常分析、经营/运营建议。没有当前消息附件时，只有用户明确说“生成报表/统计数据/分析数据”才选它。\n"
            "3. pure_chat：闲聊、开发讨论、询问你怎么实现、让你解释刚才行为、无关知识库和数据分析的问题。\n\n"
            "【重要边界】\n"
            "- 用户要求“介绍/说明/解释/讲讲”某个知识库对象或系统功能时，即使没有出现“知识库/PDF”字样，也优先选 knowledge_qa。\n"
            "- 只有完全不涉及文档资料、系统功能和表格数据的问题，才选 pure_chat。\n"
            "- “分析状态分布”“生成数据报表”属于 data_report。\n"
            "- 只有当前消息显式附加表格文件时，附件分析才走 data_report；历史临时文件不能作为自动路由依据。\n\n"
            "【示例】\n"
            "- 用户问题：详细介绍这个模块 -> {\"intent\":\"knowledge_qa\",\"confidence\":0.9,\"reason\":\"用户在询问知识库中的功能说明\"}\n"
            "- 用户问题：这个流程有哪些步骤 -> {\"intent\":\"knowledge_qa\",\"confidence\":0.9,\"reason\":\"用户在询问文档中的流程说明\"}\n"
            "- 用户问题：分析状态分布 -> {\"intent\":\"data_report\",\"confidence\":0.9,\"reason\":\"用户要求统计分析表格数据\"}\n"
            "- 用户问题：你是谁 -> {\"intent\":\"pure_chat\",\"confidence\":0.9,\"reason\":\"用户在闲聊\"}\n\n"
            "请按以下 JSON 结构输出：\n"
            f"{self.route_parser.get_format_instructions()}\n"
            "仅返回合规JSON，禁止任何多余内容！\n"
            f"用户问题：'{question}'"
        )
        response = self.llm.invoke([SystemMessage(content=route_prompt)])
        try:
            decision = self.route_parser.parse(response.content)
            if decision.intent == "pure_chat" and self._knowledge_has_relevant_hits(question):
                return "knowledge_qa"
            return decision.intent
        except:
            return "pure_chat"

    def _knowledge_has_relevant_hits(self, question: str) -> bool:
        """
        对模型路由的兜底纠偏：如果模型把短业务问题判成 pure_chat，
        但知识库向量检索能找到非空资料，就优先按知识库问答处理。

        这不是关键词路由，而是“模型判断 + 知识库召回证据”的组合判断。
        """
        normalized = question.strip()
        if len(normalized) < 2:
            return False
        knowledge_terms = ["文档", "资料", "手册", "说明", "流程", "步骤", "模块", "功能", "制度", "配置"]
        if not any(term in normalized for term in knowledge_terms):
            return False
        try:
            documents = self.vector_service.search_knowledge(normalized)
            return any(document.page_content.strip() for document in documents)
        except Exception:
            return False

    def _retrieve(self, state: RagState, config: RunnableConfig) -> RagState:
        """根据当前问题，去本地 Chroma 检索相关切片"""
        question = self._resolve_referential_question(state)
        documents = self.vector_service.search_knowledge(question)
        session_id = config.get("configurable", {}).get("thread_id", info.default_session_id)
        far_history_context = self.vector_service.search_chat_history(
            question=question,
            session_id=session_id
        )

        return {
            "documents": documents,
            "context": self._format_documents(documents),
            "far_history" :far_history_context
        }

    @staticmethod
    def _is_referential_question(question: str) -> bool:
        """
        判断用户是不是在用“上一个/刚才/之前的问题”做指代。

        RAG 检索需要可搜索的实体关键词；如果直接拿“回答我之前的问题”去搜，
        向量库当然找不到上一轮的真实主题，所以这里先识别指代句。
        """
        normalized = question.strip().lower()
        referential_patterns = [
            "之前的问题",
            "上一个问题",
            "刚才的问题",
            "刚刚的问题",
            "前面的问题",
            "那个问题",
            "这个问题",
            "继续回答",
            "重新回答",
            "再回答",
            "回答我之前",
        ]
        return any(pattern in normalized for pattern in referential_patterns)

    @staticmethod
    def _previous_user_question(messages: list) -> str:
        """
        从 LangGraph 当前消息列表里找到上一条真实用户问题。

        messages 的最后一条通常是当前 HumanMessage，所以要从倒数第二条开始找；
        遇到 AIMessage 或系统消息都跳过，直到找到上一条 HumanMessage。
        """
        for message in reversed(messages[:-1]):
            if isinstance(message, HumanMessage):
                return str(message.content).strip()
        return ""

    def _resolve_referential_question(self, state: RagState) -> str:
        """
        把“回答我之前的问题”这类指代句改写成上一条用户问题。

        第一版先做确定性改写，稳定且可解释；后续如果要支持更复杂的“这个/那个”
        跨多轮指代，再在这里补一个 LLM rewrite 节点即可。
        """
        current_question = str(state.get("question", "") or "").strip()
        if not self._is_referential_question(current_question):
            return current_question

        previous_question = self._previous_user_question(state.get("messages", []))
        if previous_question:
            return previous_question
        return current_question

    def _generate(self, state: RagState) -> RagState:
        """结合历史上下文和检索资料，由大模型生成回答，或调用聊天功能"""
        system_prompt = self._build_generation_prompt(state)
        response = self.llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
        return {"messages": [response]}

    def _build_generation_prompt(self, state: dict) -> str:
        """
        构造知识库问答/普通聊天共用的系统提示词。

        普通非流式 LangGraph 和网页流式接口都调用这里，保证人格、摘要、用户画像、
        远期记忆和知识库资料的拼接规则一致。
        """
        context = state.get("context", "")
        summary = state.get("summary", "")
        profile = state.get("profile", None)
        far_history = state.get("far_history", "")
        base_prompt = "你是本地知识库问答与数据分析助手。\n"

        if profile:
            profile_kv = [
                f"- {k}: {', '.join(v) if isinstance(v, list) else v}"
                for k, v in profile.model_dump().items() if v
            ]
            base_prompt += "【当前服务对象基本信息】：\n" + "\n".join(profile_kv) + "\n\n"

        if far_history:
            base_prompt += f"【根据历史对话向量库检索，发现用户在很久以前提及过以下相关事实，请结合参考】：\n{far_history}\n\n"

        if summary:
            base_prompt += f"【你与该用户近期对话的核心摘要】：\n{summary}\n\n"

        if context:
            system_prompt = (
                f"{base_prompt}"
                "用户当前的问题涉及专业技术，请严格、优先根据参考资料进行回答。如果资料中无法确定，请结合上下文客观说明。\n\n"
                f"【参考资料】：\n{context}"
            )
        else:
            system_prompt = (
                f"{base_prompt}"
                "用户目前正在和你闲聊或回顾状态，请完全根据上述摘要以及下述对话历史记录，用自然、友好的语气进行交谈。"
            )
        return system_prompt

    def _generate_report(self, state: RagState, config: RunnableConfig) -> RagState:
        """报表节点：优先分析当前消息附件；没有附件时才使用 docs 示例数据。"""
        question = state.get("question", "")
        session_id = config.get("configurable", {}).get("thread_id", info.default_session_id)
        analysis_file_names = state.get("analysis_file_names", [])
        session_files = self.session_file_service.resolve_files(session_id, analysis_file_names)

        if session_files:
            report_markdown = self.report_service.generate_uploaded_files_report(
                file_paths=session_files,
                question=question,
            )
            explain_prompt = (
                "你是表格数据分析助手。下面是程序已经准确统计出来的表格摘要，"
                "请基于这些统计结果进行数据解读，输出：1.总体结论；2.关键异常；"
                "3.可能原因；4.下一步建议。不要编造摘要中不存在的数字。\n\n"
                f"{report_markdown}"
            )
            try:
                response = self.llm.invoke([SystemMessage(content=explain_prompt)])
                report_markdown = report_markdown + "\n\n## 模型分析解读\n\n" + response.content
            except Exception as exc:
                report_markdown = report_markdown + f"\n\n> 模型解读失败，已保留程序统计结果。错误：{exc}"
        else:
            report_markdown = self.report_service.generate_markdown_report(question=question)

        return {
            "messages": [AIMessage(content=report_markdown)],
            "context": "",
            "documents": [],
            "far_history": ""
        }

    @staticmethod
    def _format_documents(documents: list[Document]) -> str:
        """辅助函数：拼接文本段"""
        if not documents:
            return "无相关参考资料"
        parts = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "未知来源")
            parts.append(f"资料{index}\n来源：{source}\n内容：{document.page_content}")
        return "\n\n".join(parts)

    def ask(self, question: str, session_id: str, analysis_file_names: list[str] | None = None) -> str:
        """外部调用标准接口，完整输出；analysis_file_names 表示当前消息显式附加的临时文件。"""
        config = {"configurable": {"thread_id": session_id}}
        graph_input = {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "analysis_file_names": analysis_file_names or [],
        }
        try:
            with get_mongodb_checkpointer() as checkpointer:
                graph = self.graph.compile(checkpointer=checkpointer)
                result = graph.invoke(
                    graph_input,
                    config=config,
                )
                return result["messages"][-1].content
        except Exception as persistent_exc:
            if analysis_file_names:
                session_files = self.session_file_service.resolve_files(session_id, analysis_file_names or [])
                if session_files:
                    return self.report_service.generate_uploaded_files_report(file_paths=session_files, question=question)

            try:
                graph = self.graph.compile()
                result = graph.invoke(
                    graph_input,
                    config=config,
                )
                return result["messages"][-1].content
            except Exception as stateless_exc:
                return (
                    "[服务暂不可用] 知识库问答执行失败。\n"
                    f"- 持久化执行错误：{persistent_exc}\n"
                    f"- 无状态执行错误：{stateless_exc}"
                )

    def stream_ask(self, question: str, session_id: str, analysis_file_names: list[str] | None = None):
        """
        外部调用接口，流式输出。

        这里不再依赖 LangGraph 是否透传底层模型 token，而是手写一条轻量流式链路：
        1. 读取当前会话状态，支持“回答我之前的问题”这种指代改写；
        2. 使用同一套 RouteDecision 结构化路由；
        3. 对知识库问答/闲聊回答直接调用 llm.stream()，边生成边 yield；
        4. 生成结束后把本轮 Human/AI 消息追加回 MongoDB 图状态，保留基本短期记忆。
        """
        config = {"configurable": {"thread_id": session_id}}
        analysis_file_names = analysis_file_names or []

        try:
            with get_mongodb_checkpointer() as checkpointer:
                runtime_graph = self.graph.compile(checkpointer=checkpointer)
                # 流式路径不会完整跑 LangGraph 节点，所以这里手动调用同一套记忆压缩逻辑。
                # 压缩会把旧消息滚入 summary / 远期向量记忆，只保留最近几轮进入本次上下文。
                existing_state = self.memory_manage.compact_runtime_state(
                    runtime_graph=runtime_graph,
                    config=config,
                    latest_question=question,
                )
                existing_messages = existing_state.get("messages", [])
                temp_state = {
                    "question": question,
                    "messages": existing_messages + [HumanMessage(content=question)],
                }
                resolved_question = self._resolve_referential_question(temp_state)

                if analysis_file_names:
                    route_intent: Literal["knowledge_qa", "data_report", "pure_chat"] = "data_report"
                else:
                    route_intent = self._decide_route_by_llm(resolved_question)

                if route_intent == "data_report":
                    session_files = self.session_file_service.resolve_files(session_id, analysis_file_names)
                    if session_files:
                        report_markdown = self.report_service.generate_uploaded_files_report(
                            file_paths=session_files,
                            question=resolved_question,
                        )
                    else:
                        report_markdown = self.report_service.generate_markdown_report(question=resolved_question)

                    # 程序统计结果本身已经是确定文本，先分段吐给前端，避免用户长时间看空白。
                    for start in range(0, len(report_markdown), 120):
                        yield report_markdown[start : start + 120]

                    analysis_heading = "\n\n## 模型分析解读\n\n"
                    yield analysis_heading
                    explain_prompt = (
                        "你是表格数据分析助手。下面是程序已经准确统计出来的表格摘要，"
                        "请基于这些统计结果进行数据解读，输出：1.总体结论；2.关键异常；"
                        "3.可能原因；4.下一步建议。不要编造摘要中不存在的数字。\n\n"
                        f"{report_markdown}"
                    )
                    model_parts: list[str] = []
                    try:
                        for chunk in self.llm.stream([SystemMessage(content=explain_prompt)]):
                            content = getattr(chunk, "content", "")
                            if content:
                                model_parts.append(content)
                                yield content
                    except Exception as exc:
                        error_text = f"> 模型解读失败，已保留程序统计结果。错误：{exc}"
                        model_parts.append(error_text)
                        yield error_text

                    full_report = report_markdown + analysis_heading + "".join(model_parts)

                    runtime_graph.update_state(
                        config,
                        {"messages": [HumanMessage(content=question), AIMessage(content=full_report)]},
                        as_node="generate",
                    )
                    return

                documents: list[Document] = []
                far_history_context = ""
                if route_intent == "knowledge_qa":
                    documents = self.vector_service.search_knowledge(resolved_question)
                    far_history_context = self.vector_service.search_chat_history(
                        question=resolved_question,
                        session_id=session_id,
                    )

                system_prompt = self._build_generation_prompt(
                    state={
                        "context": self._format_documents(documents) if route_intent == "knowledge_qa" else "",
                        "summary": existing_state.get("summary", ""),
                        "profile": existing_state.get("profile", None),
                        "far_history": far_history_context,
                        "messages": existing_messages,
                    }
                )
                stream_messages = [SystemMessage(content=system_prompt)] + existing_messages + [HumanMessage(content=question)]

                answer_parts: list[str] = []
                for chunk in self.llm.stream(stream_messages):
                    content = getattr(chunk, "content", "")
                    if content:
                        answer_parts.append(content)
                        yield content

                full_answer = "".join(answer_parts)
                runtime_graph.update_state(
                    config,
                    {"messages": [HumanMessage(content=question), AIMessage(content=full_answer)]},
                    as_node="generate",
                )
        except Exception as exc:
            # 流式接口不能再改 HTTP 状态码，所以把错误作为一段文本返回给前端。
            yield f"[服务暂不可用] 流式回答失败：{exc}"

    def clear_history(self, session_id: str):
        """
        一键清理：不仅清空原始消息，同时重置图状态中的 summary 和 profile，
        并物理擦除本地 Chroma 历史向量库中对应的陈年旧账。
        """
        config = {"configurable": {"thread_id": session_id}}
        with get_mongodb_checkpointer() as checkpointer:
            runtime_graph = self.graph.compile(checkpointer=checkpointer)
            existing_messages = runtime_graph.get_state(config).values.get("messages", [])

            clear_state = {
                "summary": "",
                "profile": UserProfile(),
                "far_history": "",
                "context": "",
                "documents": [],
                "analysis_file_names": []
            }

            if existing_messages:
                clear_state["messages"] = [RemoveMessage(id=m.id) for m in existing_messages]

            runtime_graph.update_state(config, clear_state, as_node="generate")

        self.vector_service.delete_chat_history(session_id=session_id)

service = RagService()
agent = service.graph.compile()
