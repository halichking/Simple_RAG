"""
知识库
"""
import os
import config_data as info
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str:str):
    """检查传入的md5字符串是否已被处理过
        return False: md5未处理；return True: 已处理，有记录"""
    if not os.path.exists(info.md5_path):
        # 文件不存在，未处理这个md5文件
        open(info.md5_path, "w", encoding="utf-8").close()
        return False
    else:
        with open(info.md5_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()  # 处理字符串前后的空格回车
                if line == md5_str:
                    return True
        return False

def save_md5(md5_str:str):
    """将传入的md5字符串记录到文件内保存"""
    with open(info.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")

def get_string_md5(input_str:str, encoding="utf-8"):
    """将传入的字符串转换为md5字符串"""
    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)
    # 创建md5对象
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()

class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在则创建，存在则跳过
        os.makedirs(info.persist_directory, exist_ok=True)

        # 向量存储的实例 Chroma向量库对象
        self.chroma = Chroma(
            collection_name=info.collection_name, # 数据库表名
            embedding_function=DashScopeEmbeddings(model=info.embedding_model_name, dashscope_api_key=info.dashscope_api_key),
            persist_directory=info.persist_directory, # 数据库本地存储文件夹
        )

        # 文本分割器的对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=info.chunk_size,       # 分隔后的文本段最大长度
            chunk_overlap=info.chunk_overlap, # 连续文本段之间的字符重叠数量
            separators=info.separators,       # 自然段落划分符号
            length_function=len,              # 使用len函数做长度统计依据
        )

    def upload_by_str(self, data: str, filename):
        """将传入的字符串进行向量化，存入向量数据库中"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > info.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks: list[str] = [data]

        # 元数据，包含：来源名词、加入时间、上传者
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "哈"
        }

        # 内容加载到向量库中
        self.chroma.add_texts(
            texts=knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # 将新增字符串的md5存入md5文件
        save_md5(md5_hex)

        return "[成功]内容已经成功存入向量库"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    for filename in os.listdir(info.docs_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(info.docs_dir, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        print(filename, service.upload_by_str(data, filename))
