import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import time, hashlib
from langchain_core.documents import Document


class DataAcquisition:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def process_webhook_data(self, markdown_content, url):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(markdown_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        final_chunks = text_splitter.split_documents(header_splits)

        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "source": url,
                "last_updated": int(time.time()),
                "chunk_id": f"{hashlib.md5(url.encode()).hexdigest()}_{i}"
            })
        return final_chunks

    def update_and_get_retriever(self, chunks, url):
        """Updates ChromaDB and returns a fresh EnsembleRetriever."""
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)

        existing = vector_db.get(where={"source": url})
        if existing["ids"]:
            vector_db.delete(ids=existing["ids"])

        if chunks:
            vector_db.add_documents(documents=chunks)

        all_data = vector_db.get(include=["documents", "metadatas"])
        if not all_data['documents']:
            print("Database is currently empty. Waiting for crawl data...")
            return vector_db.as_retriever(search_kwargs={"k": 3})
        docs = [Document(page_content=d, metadata=m) for d, m in zip(all_data['documents'], all_data['metadatas'])]

        v_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        b_retriever = BM25Retriever.from_documents(docs)

        return EnsembleRetriever(retrievers=[b_retriever, v_retriever], weights=[0.5, 0.5])