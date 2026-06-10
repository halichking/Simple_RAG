"""分层记忆管理器：短期窗口、滚动摘要、用户画像和远期向量记忆。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from RagState import RagState
from SchemaCollection import UserProfile
from vector_retriever import VectorRetrieveService


class MemoryManager:
    def __init__(self, llm, threshold: int = 8, keep_recent: int = 4):
        """
        threshold 表示触发压缩的消息数量阈值，keep_recent 表示永远保留的最近消息数。

        这里按“消息条数”控制窗口，简单、稳定、便于解释。后续如果要更精细，可以换成
        tokenizer 计数，但这版已经能避免历史无限塞进上下文。
        """
        self.llm = llm
        self.threshold = threshold
        self.keep_recent = keep_recent
        self.profile_parser = PydanticOutputParser(pydantic_object=UserProfile)
        self.vector_service = VectorRetrieveService()

    def manage_pyramid_memory(self, state: RagState, config: RunnableConfig) -> RagState:
        """LangGraph 非流式路径使用的记忆节点。"""
        messages = state.get("messages", [])
        latest_question = str(messages[-1].content) if messages else ""
        update = self.build_compaction_update(state=state, config=config, latest_question=latest_question)
        update["context"] = ""
        update["documents"] = []
        update["far_history"] = ""
        return update

    def build_compaction_update(
        self,
        state: dict[str, Any],
        config: RunnableConfig,
        latest_question: str = "",
    ) -> dict[str, Any]:
        """
        生成可写回 LangGraph state 的记忆更新。

        - 每轮都尝试从最新用户问题里抽取画像；
        - 历史消息超过阈值时，压缩旧消息为 summary；
        - 被压缩的一问一答同时沉淀到 Chroma 历史向量库；
        - 返回 RemoveMessage 命令，让 MongoDB 里不再无限堆积旧消息。
        """
        messages = list(state.get("messages", []) or [])
        current_profile = state.get("profile") or UserProfile()
        updated_profile = self.extract_profile(latest_question, current_profile) if latest_question else current_profile

        update: dict[str, Any] = {}
        if updated_profile.model_dump() != current_profile.model_dump():
            update["profile"] = updated_profile
        if len(messages) <= self.threshold:
            return update

        keep_count = max(2, self.keep_recent)
        messages_to_summarize = messages[:-keep_count]
        if not messages_to_summarize:
            return update

        summary = self._summarize_messages(
            existing_summary=state.get("summary", ""),
            messages_to_summarize=messages_to_summarize,
        )
        self._store_far_history(messages_to_summarize, config)

        removable_messages = [message for message in messages_to_summarize if getattr(message, "id", None)]
        if removable_messages:
            update["messages"] = [RemoveMessage(id=message.id) for message in removable_messages]
        update["summary"] = summary
        return update

    def compact_runtime_state(self, runtime_graph, config: RunnableConfig, latest_question: str = "") -> dict[str, Any]:
        """
        给流式接口使用的压缩入口。

        返回压缩后的 state，调用方可以立刻用短窗口 messages 组装 prompt，避免本轮仍把
        已经压缩的旧消息塞进模型上下文。
        """
        state = runtime_graph.get_state(config).values
        update = self.build_compaction_update(state=state, config=config, latest_question=latest_question)
        if self._has_meaningful_update(update):
            runtime_graph.update_state(config, update, as_node="manage_memory")
            state = runtime_graph.get_state(config).values
        return state

    def extract_profile(self, user_question: str, current_profile: UserProfile) -> UserProfile:
        """从用户当前输入里抽取稳定画像，只合并明确出现的信息。"""
        profile_prompt = (
            "你是谨慎的用户画像提取器。请从【用户刚说的这句话】中提取稳定信息。\n\n"
            f"【用户刚说的这句话】：'{user_question}'\n\n"
            "【规则】：只提取明确出现的称呼、身份、正在做的项目和偏好；"
            "没有提及的字段必须留空，禁止推测。\n\n"
            f"{self.profile_parser.get_format_instructions()}"
        )
        try:
            profile_res = self.llm.invoke([SystemMessage(content=profile_prompt)])
            extracted_profile = self.profile_parser.parse(profile_res.content)
            return current_profile.merge_with(extracted_profile)
        except Exception:
            return current_profile

    def _summarize_messages(self, existing_summary: str, messages_to_summarize: list) -> str:
        """把即将剪枝的旧消息压成一份可继续使用的滚动摘要。"""
        if existing_summary:
            summary_prompt = (
                "你是对话记忆压缩器。请在保留已有摘要的基础上，合并下面旧对话的新事实。\n\n"
                f"【已有摘要】\n{existing_summary}\n\n"
                "【要求】\n"
                "- 保留用户目标、重要约束、已完成决定、未解决问题；\n"
                "- 删除寒暄、重复内容、无长期价值的过程话；\n"
                "- 不要编造旧对话没有出现的信息；\n"
                "- 仅输出更新后的摘要正文。"
            )
        else:
            summary_prompt = (
                "你是对话记忆压缩器。请把下面旧对话压缩成后续回答仍需要的摘要。\n\n"
                "【要求】\n"
                "- 保留用户目标、重要约束、已完成决定、未解决问题；\n"
                "- 删除寒暄、重复内容、无长期价值的过程话；\n"
                "- 不要编造旧对话没有出现的信息；\n"
                "- 仅输出摘要正文。"
            )
        response = self.llm.invoke([SystemMessage(content=summary_prompt)] + messages_to_summarize)
        return str(response.content).strip()

    def _store_far_history(self, messages_to_summarize: list, config: RunnableConfig) -> None:
        """把被剪枝的一问一答写入远期向量记忆，后续按 session_id 召回。"""
        try:
            session_id = config.get("configurable", {}).get("thread_id", "local_user_001")
            for index in range(0, len(messages_to_summarize) - 1, 2):
                question_message = messages_to_summarize[index]
                answer_message = messages_to_summarize[index + 1]
                if (
                    question_message.__class__.__name__ == "HumanMessage"
                    and answer_message.__class__.__name__ == "AIMessage"
                ):
                    self.vector_service.add_chat_history_round(
                        question=str(question_message.content),
                        answer=str(answer_message.content),
                        session_id=session_id,
                    )
        except Exception as exc:
            print(f"[远期记忆沉淀失败]: {exc}")

    @staticmethod
    def _has_meaningful_update(update: dict[str, Any]) -> bool:
        """判断更新里是否有值得写回持久化状态的内容。"""
        if "summary" in update or "messages" in update:
            return True
        profile = update.get("profile")
        return bool(profile and profile.model_dump() != UserProfile().model_dump())
