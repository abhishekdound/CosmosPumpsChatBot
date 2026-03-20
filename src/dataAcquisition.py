import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import time, hashlib
from langchain_core.documents import Document
from table_to_json import TableToJSON
from univeral_table_parser import UniversalTableParser
from bs4 import BeautifulSoup
from llm_table_to_json import LLMTableToJson
from llm import llm
from image_analyzer import ImageAnalyzer

from rag_processor import RAGProcessor


class DataAcquisition:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_db =Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)

    def extract_raw_tables(self,html):
        soup = BeautifulSoup(html, "html.parser")
        return [t.get_text(" ", strip=True) for t in soup.find_all("table")]

    def json_to_sentences(self,table_json, context=""):
        sentences = []

        if not isinstance(table_json, dict):
            return []

        for metric, values in table_json.items():
            if not isinstance(values, dict): continue
            for col, val in values.items():
                s = f"{metric} in {col} is {val}."
                if context:
                    s = f"{context}: {s}"
                sentences.append(s)
        return sentences


    def process_webhook_data(self, markdown_content,html_content, url):
        parser = UniversalTableParser()
        json_converter = TableToJSON()
        image_analyze=ImageAnalyzer()
        table_sentences = []
        image_descriptions = []
        if html_content:
            tables = parser.parse_html(html_content)


            for t in tables:
                table_json = json_converter.convert(t)

                sentences = self.json_to_sentences(table_json, context=url)

                table_sentences.extend(sentences)

            if len(table_sentences) < 3 and len(tables) > 0 and html_content:

                raw_tables = self.extract_raw_tables(html_content)

                for raw in raw_tables:
                    try:
                        table_json = LLMTableToJson(llm, raw[:2000])

                        sentences = self.json_to_sentences(table_json, context=url)
                        table_sentences.extend(sentences)

                    except Exception as e:
                        print("LLM fallback failed:", e)


            image_descriptions=image_analyze.process_all_images(html_content,url)


        if not table_sentences and markdown_content:
            tables = parser.parse_markdown(markdown_content)

            for t in tables:
                table_json = json_converter.convert(t)

                sentences = self.json_to_sentences(table_json, context=url)

                table_sentences.extend(sentences)

        table_sentences = list(set(table_sentences))
        image_descriptions = list(set(image_descriptions))
        enriched_content = (
                (markdown_content or "") +
                "\n" +
                "\n".join(table_sentences) +
                "\n" +
                "\n".join(image_descriptions)
        )
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(enriched_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
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
                weights=[0.4, 0.6]
            )
        else:
            return v_retriever

    async def update_retriever_from_image_bytes(self, image_bytes, source="user_image"):
        """
        Processes image bytes → description → chunk → update retriever
        """


        image_analyzer = ImageAnalyzer()
        rag_processor = RAGProcessor()

        extracted_text = await image_analyzer.describe_image_with_vlm(image_bytes)

        if not extracted_text or not extracted_text.strip():
            print("VLM processing failed or returned empty content")
            return self.vector_db.as_retriever()

        print("\n=== VLM OUTPUT (Markdown/Text) ===\n", extracted_text)

        chunks = rag_processor.chunk_content(
            content=extracted_text,
            source=f"{source}_{time.time()}"
        )

        vector_db = self.vector_db


        vector_db.add_documents(chunks)

        all_data = vector_db.get(include=["documents", "metadatas"])

        print(f"[IMAGE] Total Chunks: {len(all_data['ids'])}")

        if not all_data['documents']:
            return vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8}
            )

        docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(all_data['documents'], all_data['metadatas'])
        ]

        v_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 40}
        )

        b_retriever = BM25Retriever.from_documents(docs)
        b_retriever.k = 10

        return EnsembleRetriever(
            retrievers=[b_retriever, v_retriever],
            weights=[0.4, 0.6]
        )

    async def update_retriever_from_document_bytes(self, doc_bytes, mime_type, filename):
        if mime_type == "application/pdf":
            # use pypdf or pdfplumber to extract text
            pass
        elif mime_type == "text/plain":
            text = doc_bytes.decode("utf-8")
            # chunk and embed text
            pass
        elif "wordprocessingml" in mime_type:
            # use python-docx to extract text
            pass


if __name__=='__main__':
    da = DataAcquisition()
    test_chunk = Document(
        page_content="Virat Kohli is an Indian cricketer.",
        metadata={"source": "test_url"}
    )
    da.update_and_get_retriever([test_chunk], 'test_url')
