"""
MongoDB线程持久化记忆存储
"""
from langgraph.checkpoint.mongodb import MongoDBSaver
import ConfigData as info

MONGODB_URL = info.mongodb_url

def get_mongodb_checkpointer() -> MongoDBSaver:
    """
    直接获取官方原生的 MongoDBSaver 实例。
    MongoDBSaver 内部已经完美实现了原生上下文协议，
    在外部直接使用 with 即可自动安全管理 Docker 连接开关。
    """
    return MongoDBSaver.from_conn_string(
        host_string=MONGODB_URL,
        client_kwargs={"serverSelectionTimeoutMS": 3000}    # 3秒连接超时
    )