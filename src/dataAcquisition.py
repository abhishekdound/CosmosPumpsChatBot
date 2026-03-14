from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from llm import llm


class DataAcquisition:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def chunks(self, url="https://en.wikipedia.org/wiki/Virat_Kohli"):

        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()

        fact_prompt = ChatPromptTemplate.from_template("""
    Extract factual statements from the text below.
    Convert tables, statistics, and structured data into clear sentences.

    Text:
    {text}

    Return only the extracted facts.
    """)

        fact_chain = fact_prompt | llm
        fact_docs = []

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
            doc.metadata["source"] = url

        responses = fact_chain.batch(
            [{"text": doc.page_content} for doc in documents]
        )

        for response in responses:

            facts = response.content.split("\n")

            for fact in facts:
                fact = fact.strip("-•0123456789. ").strip()
                if fact and len(fact) < 300:
                    fact_docs.append(
                        Document(
                            page_content=fact,
                            metadata={"source": url, "type": "fact"}
                        )
                    )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        for c in chunks:
            c.metadata["source"] = url
            c.metadata["type"] = "chunk"

        return chunks + fact_docs

    def save_vector_DB(self, chunks):

        vector_db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory="./chroma_db"
        )

        vector_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 50, "fetch_k": 120}
        )

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 20

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )

    def load_vector_DB(self):

        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        vector_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 50, "fetch_k": 120}
        )

        docs = vector_db.get(include=["documents", "metadatas"])

        bm25_docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(docs["documents"], docs["metadatas"])
        ]

        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 20

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )