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
        self.vector_db =Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)

    def process_webhook_data(self, markdown_content, url):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(markdown_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        final_chunks = text_splitter.split_documents(header_splits)

        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "source": url,
                "header": chunk.metadata.get("Header 1", ""),
                "last_updated": int(time.time()),
                "chunk_id": f"{hashlib.md5(url.encode()).hexdigest()}_{i}"
            })
        return final_chunks

    def update_and_get_retriever(self, chunks, url):
        """Updates ChromaDB and returns a fresh EnsembleRetriever."""
        vector_db = self.vector_db

        existing = vector_db.get(where={"source": url})
        if existing["ids"]:
            vector_db.delete(ids=existing["ids"])

        if chunks:
            vector_db.add_documents(documents=chunks)

        all_data = vector_db.get(include=["documents", "metadatas"])

        print(f"Total Chunks found: {len(all_data['ids'])}")

        if len(all_data['ids']) > 0:
            print(f"First Chunk Sample: {all_data['documents'][0][:100]}...")
        else:
            print("DATABASE IS TOTALLY EMPTY")
        if not all_data['documents']:
            print("Database is currently empty.Waiting for crawl data")
            return vector_db.as_retriever(
                                            search_type="mmr",
                                            search_kwargs={"k":8, "fetch_k":20}
                                        )
        docs = [Document(page_content=d, metadata=m) for d, m in zip(all_data['documents'], all_data['metadatas'])]

        for doc in docs:
            if "height" in doc.page_content.lower():
                print(doc.page_content)
        v_retriever = vector_db.as_retriever(
                                                search_type="mmr",
                                                search_kwargs={"k":8, "fetch_k":20}
                                            )
        if len(docs) < 2000:
            b_retriever = BM25Retriever.from_documents(docs)
            b_retriever.k = 6
        else:
            b_retriever = None

        if b_retriever:
            return EnsembleRetriever(
                retrievers=[b_retriever, v_retriever],
                weights=[0.4, 0.6]
            )
        else:
            return v_retriever


if __name__=='__main__':
    da = DataAcquisition()
    test_chunk = Document(
        page_content="Virat Kohli is an Indian cricketer.",
        metadata={"source": "test_url"}
    )
    da.update_and_get_retriever([test_chunk], 'test_url')
