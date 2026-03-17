import time, hashlib
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGProcessor:

    def chunk_content(self, content, source):

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
        )

        header_splits = markdown_splitter.split_text(content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        final_chunks = text_splitter.split_documents(header_splits)

        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "source": source,
                "type": "image" if "Image at" in chunk.page_content else "text",
                "chunk_id": f"{hashlib.md5(source.encode()).hexdigest()}_{i}",
                "last_updated": int(time.time())
            })

        return final_chunks