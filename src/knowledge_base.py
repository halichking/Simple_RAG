"""
知识库 knowledge_base
"""
import os
import hashlib
from pathlib import Path
import ConfigData as info
from report_service import read_xlsx_rows
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

MD5_PATH = info.md5_path
PERSIST_DIR = info.persist_directory
COLLECTION_NAME = info.knowledge_collection_name
MODEL_NAME = info.embedding_model_name
API_KEY = info.dashscope_api_key
CHUNK_SIZE = info.chunk_size
CHUNK_OVERLAP = info.chunk_overlap
SEPARATORS = info.separators
MAX_SPLIT_CHAR_NUMBER = info.max_split_char_number
OPERATOR_NAME = "UserName"

def check_md5(md5_str:str) -> bool:
    """
    检查传入的md5字符串是否已被处理
    :param md5_str:
    :return: False -> 文件未处理
           : True -> 已处理
    """
    if not os.path.exists(MD5_PATH):
        # 保存md5的文件不存在，创立文件夹并返回False
        os.makedirs(os.path.dirname(MD5_PATH), exist_ok=True)
        open(MD5_PATH, "w", encoding="utf-8").close()
        return False
    else:
        with open(MD5_PATH, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line == md5_str:
                    return True
        return False

def save_md5(md5_str:str):
    """将传入的md5字符串记录到文件中保存"""
    with open(MD5_PATH, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")

def get_string_md5(input_str:str, encoding="utf-8") -> str:
    """将传入的字符串转换为md5字符串"""
    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)
    #创建md5对象
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()

class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(PERSIST_DIR, exist_ok=True)

        # 向量存储实例 Chroma 向量库对象
        self.chroma = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=DashScopeEmbeddings(model=MODEL_NAME, dashscope_api_key=API_KEY),
            persist_directory=PERSIST_DIR,
        )
        # 文本分割器对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len
        )

    def upload_by_str(self, data:str, filename):
        """将传入字符串向量化，存入向量数据库"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > MAX_SPLIT_CHAR_NUMBER:
            knowledge_chunks:list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks:list[str] = [data]

        # 元数据：文件名，上传时间，上传者
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": OPERATOR_NAME
        }
        # 加载到数据库
        self.chroma.add_texts(
            texts=knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )
        # 将新增字符串的md5存入md5.txt
        save_md5(md5_hex)

        return "[成功]内容已经成功存入向量库"

    def upload_by_file(self, file_path:str) -> str:
        """
        判断文件类型是txt、pdf还是xlsx。

        注意：Excel/CSV 表格属于结构化数据，默认不直接混入普通 RAG 向量库；
        表格分析统一交给 ReportService，避免表格明细污染文档问答检索结果。
        :param file_path: 文件的绝对/相对路径
        """
        if not os.path.exists(file_path):
            return f"[错误]文件路径不存在：{file_path}"

        # 获得文件本身名字
        filename = os.path.basename(file_path)

        # 1.如果是txt
        if file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
            return self.upload_by_str(data, filename)

        # 2.如果是pdf
        elif file_path.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                # loader.load()会把每一页都读成一个 Document 对象
                pages = loader.load()
                # 将每一页的 page_content 纯文本用换行符拼接，变为长文本
                data = "\n\n".join([page.page_content for page in pages])
                return self.upload_by_str(data, filename)
            except Exception as e:
                return f"[失败]读取PDF文件出错：{str(e)}"

        # 3.如果是xlsx，只验证可读取，不写入知识库向量集合
        elif file_path.lower().endswith(".xlsx"):
            try:
                rows = read_xlsx_rows(os.path.abspath(file_path))
                data_rows = max(0, len(rows) - 2)
                return f"[跳过]已识别Excel结构化数据：{filename}，共读取约 {data_rows} 行；请使用 ReportService 生成分析报表"
            except Exception as e:
                return f"[失败]读取Excel文件出错：{str(e)}"

        else: return f"[跳过]不支持的文件格式，当前仅支持 .txt、.pdf 和 .xlsx"

    def upload_docs_dir(self, docs_dir: str | Path | None = None) -> list[str]:
        """
        批量扫描 docs 目录并处理支持的文件。

        - PDF/TXT：提取文本后写入知识库向量库，用于本地资料问答；
        - XLSX：只验证结构化表格可读取，不写入普通问答向量库，避免报表数据污染 PDF 问答；
        - 其他格式：跳过并返回说明，方便网页端展示处理结果。
        """
        target_dir = Path(docs_dir) if docs_dir else info.docs_dir
        if not target_dir.exists():
            return [f"[错误]文档目录不存在：{target_dir}"]

        results: list[str] = []
        supported_suffixes = {".txt", ".pdf", ".xlsx"}
        for file_path in sorted(target_dir.iterdir()):
            if file_path.name.startswith("~$") or not file_path.is_file():
                continue
            if file_path.suffix.lower() not in supported_suffixes:
                results.append(f"[跳过]不支持的文件格式：{file_path.name}")
                continue

            # 每个文件独立处理，某个文件失败时不影响后续文件继续入库。
            result = self.upload_by_file(str(file_path))
            results.append(f"{file_path.name}: {result}")

        if not results:
            return [f"[提示]目录中没有可处理文件：{target_dir}"]
        return results

if __name__ == '__main__':
    service = KnowledgeBaseService()

    print("开始批量同步 docs 目录...")
    for item in service.upload_docs_dir():
        print(item)
