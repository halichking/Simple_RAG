import config_data as info
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=info.collection_name,
            embedding_function=self.embedding,
            persist_directory=info.persist_directory,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": info.similarity_threshold})

if __name__ == '__main__':
    retriever = VectorStoreService(DashScopeEmbeddings(model=info.embedding_model_name, dashscope_api_key=info.dashscope_api_key)).get_retriever()

    res = retriever.invoke("我的体重138斤，尺码推荐")
    print(res)
