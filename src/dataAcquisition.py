from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


class DataAcquisition:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def chunks(self, url="https://en.wikipedia.org/wiki/Virat_Kohli"):

        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()

        def normalize_tables(text):
            lines = text.split("\n")
            new_lines = []

            for line in lines:
                if "|" in line:
                    line = line.replace("|", " ")
                new_lines.append(line)

            return "\n".join(new_lines)

        for doc in documents:
            doc.page_content = normalize_tables(doc.page_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        return splitter.split_documents(documents)

    def save_vector_DB(self, chunks):

        vector_db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory="./chroma_db"
        )

        vector_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30}
        )

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

    def load_vector_DB(self):

        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        vector_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30}
        )

        chunks = self.chunks()
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )