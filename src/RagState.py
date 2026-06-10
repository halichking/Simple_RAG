from langgraph.graph import MessagesState
from langchain_core.documents import Document
from SchemaCollection import UserProfile

class RagState(MessagesState):
    question: str               # 用户输入的纯文本问题
    documents: list[Document]   # 向量库检索出来的原始分段切片列表
    context: str                # 格式化后拼接给大模型的参考资料字符串
    summary: str                # 对话历史摘要
    profile: UserProfile        # 用户画像
    far_history: str        # 远期记忆
    # 当前这条用户消息显式附加的临时分析文件名。
    # 注意：这里不是“会话目录里的全部文件”，否则旧文件会反复参与后续分析。
    analysis_file_names: list[str]
