import os
import io
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

import fitz
import pdfplumber
from docx import Document as DocxDocument


class DataAcquisition:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _get_session_db(self, session_id: str) -> Chroma:
            """In-memory Chroma collection per session — auto wiped when cleared."""
            return Chroma(
                collection_name=f"session_{session_id}",
                embedding_function=self.embeddings
            )

    def clear_session(self, session_id: str):
        """Wipes all chunks from session DB."""
        try:
            session_db = self._get_session_db(session_id)
            all_ids = session_db.get()["ids"]
            if all_ids:
                session_db.delete(ids=all_ids)
                print(f"[SESSION] Cleared {len(all_ids)} chunks for {session_id}")
            else:
                print(f"[SESSION] Nothing to clear for {session_id}")
        except Exception as e:
            print(f"[SESSION] Clear failed: {e}")

        try:
            crawl_db = self._get_session_db(f"crawl_{session_id}")
            ids = crawl_db.get()["ids"]
            if ids:
                crawl_db.delete(ids=ids)
                print(f"[CRAWL] Cleared {len(ids)} crawl chunks for {session_id}")
        except Exception as e:
            print(f"[CRAWL] Crawl clear failed: {e}")

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

    def build_retriever(self, session_id: str = None):
        """Shared helper — builds EnsembleRetriever from current Chroma state."""

        all_docs = []
        crawl_docs = []
        if session_id:
            crawl_db = self._get_session_db(f"crawl_{session_id}")
            crawl_data = crawl_db.get(include=["documents", "metadatas"])
            if crawl_data["documents"]:
                crawl_docs = [
                    Document(page_content=d, metadata=m)
                    for d, m in zip(crawl_data["documents"], crawl_data["metadatas"])
                ]
                all_docs += crawl_docs

        session_docs = []
        if session_id:
            session_db = self._get_session_db(session_id)
            session_data = session_db.get(include=["documents", "metadatas"])
            if session_data["documents"]:
                session_docs = [
                    Document(page_content=d, metadata=m)
                    for d, m in zip(session_data["documents"], session_data["metadatas"])
                ]
                all_docs += session_docs

        print(f"[RETRIEVER] Crawl: {len(crawl_docs)} | Session: {len(session_docs)}")

        if not all_docs:
            empty_db = self._get_session_db("empty")
            return empty_db.as_retriever(search_type="mmr", search_kwargs={"k": 8})

        retrievers = []
        weights = []

        b_retriever = BM25Retriever.from_documents(all_docs)
        b_retriever.k = 10
        retrievers.append(b_retriever)
        weights.append(0.2)

        if crawl_docs and session_id:
            v_crawl = self._get_session_db(f"crawl_{session_id}").as_retriever(
                search_type="mmr",
                search_kwargs={"k": 15, "fetch_k": 40}
            )
            retrievers.append(v_crawl)
            weights.append(0.2)

        if session_docs:
            session_db = self._get_session_db(session_id)
            v_session = session_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 15, "fetch_k": 40}
            )
            retrievers.append(v_session)
            weights.append(0.6)
        if len(retrievers) == 1:
            return retrievers[0]

        return EnsembleRetriever(retrievers=retrievers, weights=weights)

    def extract_text_from_pdf(self, doc_bytes: bytes) -> str:
        """Extract all text from PDF using pdfplumber."""
        text_parts = []
        with pdfplumber.open(io.BytesIO(doc_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()

                filtered_page = page
                if tables:
                    for table_obj in page.find_tables():
                        filtered_page = filtered_page.outside_bbox(table_obj.bbox)

                text = filtered_page.extract_text()
                if text:
                    text_parts.append(f"## Page {page_num}\n{text}")

                for table in tables:
                    if not table: continue
                    headers = table[0]
                    md_table = "| " + " | ".join(str(h or "") for h in headers) + " |\n"
                    md_table += "| " + " | ".join("---" for _ in headers) + " |\n"
                    for row in table[1:]:
                        md_table += "| " + " | ".join(str(c or "") for c in row) + " |\n"
                    text_parts.append(md_table)

        return "\n\n".join(text_parts)

    async def _extract_images_from_pdf(self, doc_bytes: bytes, filename: str) -> list[str]:
        """Extract embedded images from PDF and run VLM on each."""
        image_analyzer = ImageAnalyzer()
        image_descriptions = []

        pdf = fitz.open(stream=doc_bytes, filetype="pdf")

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]

                    if len(image_bytes) < 5000:
                        continue

                    print(f"PDF Processing image {img_index + 1} on page {page_num + 1}...")
                    extracted_text = await image_analyzer.describe_image_with_vlm(image_bytes)

                    if extracted_text and "NO_TEXT_FOUND" not in extracted_text:
                        entry = (
                            f" Image from {filename} "
                            f"(Page {page_num + 1}, Image {img_index + 1}) ---\n"
                            f"{extracted_text}"
                        )
                        image_descriptions.append(entry)

                except Exception as e:
                    print(f"[PDF] Image extraction error: {e}")
                    continue

        pdf.close()
        return image_descriptions

    def extract_text_from_docx(self, doc_bytes: bytes) -> str:

        docx = DocxDocument(io.BytesIO(doc_bytes))
        parts = []

        for para in docx.paragraphs:
            text = para.text.strip()
            if not text: continue
            style = para.style.name
            if "Heading 1" in style:
                parts.append(f"# {text}")
            elif "Heading 2" in style:
                parts.append(f"## {text}")
            elif "Heading 3" in style:
                parts.append(f"### {text}")
            else:
                parts.append(text)

        for table in docx.tables:
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            if not rows: continue
            md = "| " + " | ".join(rows[0]) + " |\n"
            md += "| " + " | ".join("---" for _ in rows[0]) + " |\n"
            for row in rows[1:]:
                md += "| " + " | ".join(row) + " |\n"
            parts.append(md)

        return "\n\n".join(parts)

    async def _extract_images_from_docx(self, doc_bytes: bytes, filename: str) -> list[str]:
        """Extract embedded images from DOCX and run VLM on each."""
        image_analyzer = ImageAnalyzer()
        image_descriptions = []

        docx = DocxDocument(io.BytesIO(doc_bytes))

        for rel in docx.part.rels.values():
            if "image" not in rel.reltype:
                continue
            try:
                image_part = rel.target_part
                image_bytes = image_part.blob

                if len(image_bytes) < 5000:
                    continue

                print(f"[DOCX] Processing embedded image from {filename}...")
                extracted_text = await image_analyzer.describe_image_with_vlm(image_bytes)

                if extracted_text and "NO_TEXT_FOUND" not in extracted_text:
                    entry = (
                        f"--- Image from {filename} ---\n"
                        f"{extracted_text}"
                    )
                    image_descriptions.append(entry)

            except Exception as e:
                print(f"[DOCX] Image extraction error: {e}")
                continue

        return image_descriptions

    async def update_retriever_from_document_bytes(
            self, doc_bytes: bytes, mime_type: str, filename: str,session_id:str
    ):
        """
        PDF / DOCX / TXT → extract text + images → chunk → embed → retriever
        """
        rag_processor = RAGProcessor()

        text_content = ""
        image_descriptions = []

        if mime_type == "application/pdf":
            print(f"[DOC] Extracting text from PDF: {filename}")
            text_content = self.extract_text_from_pdf(doc_bytes)

            print(f"[DOC] Extracting images from PDF: {filename}")
            image_descriptions = await self._extract_images_from_pdf(doc_bytes, filename)

        elif "wordprocessingml" in mime_type:
            print(f"[DOC] Extracting text from DOCX: {filename}")
            text_content = self.extract_text_from_docx(doc_bytes)

            print(f"[DOC] Extracting images from DOCX: {filename}")
            image_descriptions = await self._extract_images_from_docx(doc_bytes, filename)

        elif mime_type == "text/plain":
            print(f"[DOC] Reading TXT: {filename}")
            text_content = doc_bytes.decode("utf-8", errors="ignore")

        else:
            print(f"[DOC] Unsupported mime type: {mime_type}")
            return self._get_session_db("empty").as_retriever()

        if not text_content and not image_descriptions:
            print("[DOC] Nothing extracted from document.")
            return self._get_session_db("empty").as_retriever()

        full_content = text_content
        if image_descriptions:
            full_content += "\n\n" + "\n\n".join(image_descriptions)

        print(f"[DOC] Total content length: {len(full_content)} chars")
        print(f"[DOC] Images extracted: {len(image_descriptions)}")

        source_id = f"{filename}_{int(time.time())}"
        chunks = rag_processor.chunk_content(
            content=full_content,
            source=source_id
        )

        session_db = self._get_session_db(session_id)
        session_db.add_documents(chunks)
        print(f"[DOC] Added {len(chunks)} chunks to session {session_id}")

        return self.build_retriever(session_id)


    async def process_webhook_data(self, markdown_content, html_content, url):
        parser = UniversalTableParser()
        json_converter = TableToJSON()
        image_analyze = ImageAnalyzer()
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

            image_descriptions =await image_analyze.process_all_images(html_content, url)

        if not table_sentences and markdown_content:
            tables = parser.parse_markdown(markdown_content)
            for t in tables:
                table_json = json_converter.convert(t)
                sentences = self.json_to_sentences(table_json, context=url)
                table_sentences.extend(sentences)

        table_sentences = list(set(table_sentences))
        image_descriptions = list(set(image_descriptions))

        enriched_content = (
            (markdown_content or "") + "\n" +
            "\n".join(table_sentences) + "\n" +
            "\n".join(image_descriptions)
        )

        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
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

    def update_and_get_retriever(self, chunks, url, session_id: str):
        crawl_key = f"crawl_{session_id}"
        crawl_db = self._get_session_db(crawl_key)

        existing = crawl_db.get(where={"source": url})
        if existing["ids"]:
            crawl_db.delete(ids=existing["ids"])

        if chunks:
            crawl_db.add_documents(documents=chunks)

        return self.build_retriever(session_id=session_id)

    async def update_retriever_from_image_bytes(self, image_bytes,session_id:str, source="user_image"):
        image_analyzer = ImageAnalyzer()
        rag_processor = RAGProcessor()

        extracted_text = await image_analyzer.describe_image_with_vlm(image_bytes)

        if not extracted_text or not extracted_text.strip():
            print("VLM processing failed or returned empty content")
            return "",self._get_session_db("empty").as_retriever()

        print("\n=== VLM OUTPUT ===\n", extracted_text)

        chunks = rag_processor.chunk_content(
            content=extracted_text,
            source=f"{source}_{time.time()}"
        )

        session_db = self._get_session_db(session_id)
        session_db.add_documents(chunks)
        print(f"[IMAGE] Added {len(chunks)} chunks to session {session_id}")
        return extracted_text,self.build_retriever(session_id)

    def copy_crawl_to_session(self, from_session: str, to_session: str):
        try:
            src_db = self._get_session_db(f"crawl_{from_session}")
            src_data = src_db.get(include=["documents", "metadatas"])

            if not src_data["documents"]:
                print(f"[CRAWL] Nothing to copy from {from_session}")
                return

            dst_db = self._get_session_db(f"crawl_{to_session}")
            docs = [
                Document(page_content=d, metadata=m)
                for d, m in zip(src_data["documents"], src_data["metadatas"])
            ]
            dst_db.add_documents(docs)
            print(f"[CRAWL] Copied {len(docs)} chunks from {from_session} to {to_session}")

        except Exception as e:
            print(f"[CRAWL] Copy failed: {e}")


if __name__=='__main__':
    da = DataAcquisition()
    test_chunk = Document(
        page_content="Virat Kohli is an Indian cricketer.",
        metadata={"source": "test_url"}
    )
    da.update_and_get_retriever([test_chunk], 'test_url')
