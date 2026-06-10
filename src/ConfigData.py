"""读取项目根目录的 config_data.yml 和 .env。"""

from __future__ import annotations
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config_data.yml"

load_dotenv(PROJECT_ROOT / ".env")


def _read_yaml() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


config = _read_yaml()

docs_dir = _path(config["paths"]["docs_dir"])
data_dir = _path(config["paths"]["data_dir"])
vector_store_dir = _path(config["paths"]["vector_store_dir"])
chat_history_dir = _path(config["paths"]["chat_history_dir"])
session_upload_dir = _path(config["paths"].get("session_upload_dir", "data/session_uploads"))
md5_path = str(_path(config["paths"]["md5_path"]))

embedding_model_name = config["models"]["embedding_model"]
chat_model_base_url = config["models"]["aliyun_base_url"]
chat_model_name = config["models"]["aliyun_chat_model"]

knowledge_collection_name = config["vector_store"]["knowledge_collection_name"]
history_collection_name = config["vector_store"]["history_collection_name"]
persist_directory = str(vector_store_dir)

mongodb_url = config["mongodb_store"]["mongodb_url"]
db_name = config["mongodb_store"]["db_name"]

chunk_size = config["splitter"]["chunk_size"]
chunk_overlap = config["splitter"]["chunk_overlap"]
max_split_char_number = config["splitter"]["max_split_char_number"]
separators = config["splitter"]["separators"]

retriever_top_k = config["retriever"]["top_k"]
similarity_threshold = config["retriever"]["similarity_threshold"]
default_session_id = config["session"]["default_session_id"]
report_duplicate_strategy = config.get("report", {}).get("duplicate_strategy", "md5")

chat_model_api_key = os.getenv("ALIYUN_API_KEY")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmith_project = os.getenv("LANGSMITH_PROJECT")


def has_dashscope_api_key() -> bool:
    return bool(dashscope_api_key)


def ensure_dirs() -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    chat_history_dir.mkdir(parents=True, exist_ok=True)
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    Path(md5_path).parent.mkdir(parents=True, exist_ok=True)


ensure_dirs()


if __name__ == "__main__":
    print(f"docs_dir: {docs_dir}")
    print(f"md5_path: {md5_path}")
    print(f"vector_store_dir: {vector_store_dir}")
    print(f"chat_history_dir: {chat_history_dir}")
    print(f"embedding_model_name: {embedding_model_name}")
    print(f"chat_model_name: {chat_model_name}")
    print(f"has_dashscope_api_key: {has_dashscope_api_key()}")
