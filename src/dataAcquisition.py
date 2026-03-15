import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document


class DataAcquisition:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def chunks(self, url="https://en.wikipedia.org/wiki/Virat_Kohli"):
        loader = FireCrawlLoader(
            api_key=os.getenv('FIRECRAWL_API_KEY'),
            url=url,
            mode="crawl",
            params={
                "formats": ["markdown", "html"],
                "waitFor": 2000,
                "onlyMainContent": False,
                "limit":10
            }
        )
        documents = loader.load()
        content = "\n\n".join([doc.page_content for doc in documents])
        content = content.replace("|", " ")

        search_term = "height"
        if search_term.lower() in content.lower():
            print(f"✅ SUCCESS: '{search_term}' found in raw scrape.")
        else:
            print(f"❌ WARNING: '{search_term}' NOT FOUND. FireCrawl may be truncating the page.")

        if not content.startswith("#"):
            content = f"# Introduction\n{content}"

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        header_splits = markdown_splitter.split_text(content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400
        )
        final_chunks = text_splitter.split_documents(header_splits)

        for i, chunk in enumerate(final_chunks):
            subject = chunk.metadata.get("Header 1") or chunk.metadata.get("og:title") or "Article"
            section = chunk.metadata.get("Header 2") or "Main Content"

            chunk.page_content = f"Source: {subject} ({section})\n{chunk.page_content}"

            chunk.metadata.update({
                "chunk_id": i,
                "source": f"{subject} - {section}"
            })

        return final_chunks

    def save_vector_DB(self, chunks):

        vector_db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory="./src/chroma_db",
            collection_metadata={"hnsw:space": "cosine"}
        )

        vector_retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7}
        )

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 7

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.7, 0.3]
        )

    def load_vector_DB(self):

        vector_db = Chroma(
            persist_directory="./src/chroma_db",
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

        vector_retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7}
        )

        docs = vector_db.get(include=["documents", "metadatas"])

        bm25_docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(docs["documents"], docs["metadatas"])
        ]

        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 7

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.7, 0.3]
        )