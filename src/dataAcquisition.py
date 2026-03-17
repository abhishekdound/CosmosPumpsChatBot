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

    # def expand_table_rows(self, text):
    #     lines = text.split("\n")
    #     expanded = []
    #
    #     for line in lines:
    #         if "|" in line:
    #             parts = [p.strip() for p in line.split("|") if p.strip()]
    #
    #             if len(parts) == 2:
    #                 key, value = parts
    #                 expanded.append(f"{key} : {value}.")
    #             elif len(parts) > 2:
    #                 # row style table
    #                 key = parts[0]
    #                 for value in parts[1:]:
    #                     expanded.append(f"{key} : {value}.")
    #         else:
    #             expanded.append(line)
    #
    #     return "\n".join(expanded)


    def process_webhook_data(self, markdown_content, url):
        # markdown_content = self.expand_table_rows(markdown_content)
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(markdown_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
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
            return vector_db.as_retriever(
                                            search_type="mmr",
                                            search_kwargs={"k":8}
                                        )
        docs = [Document(page_content=d, metadata=m) for d, m in zip(all_data['documents'], all_data['metadatas'])]

        for doc in docs:
            if "height" in doc.page_content.lower():
                print("yes present")
        v_retriever = vector_db.as_retriever(
                                                search_type="mmr",
                                                search_kwargs={"k":15, "fetch_k":40}
                                            )
        b_retriever = BM25Retriever.from_documents(docs)
        b_retriever.k = 10

        if b_retriever:
            return EnsembleRetriever(
                retrievers=[b_retriever, v_retriever],
                weights=[0.55, 0.45]
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
