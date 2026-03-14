from langchain_community.document_loaders import FireCrawlLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


class DataAcquisition:
    def __init__(self):
        self.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def chunks(self,url="https://en.wikipedia.org/wiki/Virat_Kohli"):
        loader=FireCrawlLoader(url=url,mode="crawl")
        documents=loader.load()
        splitter=RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        return chunks
    def save_vector_DB(self,chunks):
        vector_db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory="./chroma_db"
        )
        return vector_db.as_retriever(search_kwargs={"k": 6})

    def load_vector_DB(self):
        vector_db=Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        return vector_db.as_retriever(search_kwargs={"k": 6})


