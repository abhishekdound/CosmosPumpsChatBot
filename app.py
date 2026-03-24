import asyncio
import os
import time
import httpx
from bs4 import BeautifulSoup
import markdownify

import chainlit as cl
import requests
import re
from src.llmHelper import graph
import src.webHookListner as webHookListner


from src.dataAcquisition import DataAcquisition
da=DataAcquisition()


def is_url(text: str) -> bool:
    """Detect if message is a URL."""
    text = text.strip()
    url_pattern = re.compile(
        r'^(https?://)'           
        r'([a-zA-Z0-9.-]+)'       
        r'(\.[a-zA-Z]{2,})'       
        r'(/[^\s]*)?$'
    )
    return bool(url_pattern.match(text))

@cl.on_chat_start
async def start():
    thread_id = str(time.time())
    cl.user_session.set("thread_id", thread_id)

    async def on_crawl_complete(session_id: str):
        await cl.Message(
            content=" *Crawl complete* Knowledge base updated. Ask me anything!"
        ).send()

    webHookListner.register_session(thread_id)

    await cl.Message(
        content=""" **Welcome to CosmosPumps AI!**

    I can help you with:
    -  Web Knowledge — Add any website as a knowledge source
    -  Documents — Upload PDF, DOCX, or TXT files
    -  Images — Upload images for analysis and extraction
    -  Questions — Ask anything from your uploaded content

    Get started by using one of the options below or just type your question!""",
        actions=[
            cl.Action(
                name="add_website",
                label=" Add Website",
                payload={"type": "website"}
            ),
            cl.Action(
                name="upload_doc",
                label=" Upload Document",
                payload={"type": "document"}
            ),
            cl.Action(
                name="upload_image",
                label=" Upload Image",
                payload={"type": "image"}
            ),
            cl.Action(
                name="how_to_use",
                label=" How to Use",
                payload={"type": "help"}
            ),
        ]
    ).send()



@cl.action_callback("add_website")
async def handle_add_website(action: cl.Action):
    await cl.Message(
        content=" Please enter the website URL you want to crawl:\n*(e.g. https://example.com)*"
    ).send()


@cl.action_callback("upload_doc")
async def handle_upload_doc(action: cl.Action):
    await cl.Message(
        content=" Please upload your document using the  attachment button in the input box below.\n\n*Supported formats: PDF, DOCX, TXT*"
    ).send()


@cl.action_callback("upload_image")
async def handle_upload_image(action: cl.Action):
    await cl.Message(
        content=" Please upload your image using the  attachment button in the input box below.\n\n*Supported formats: PNG, JPG, JPEG, WEBP*"
    ).send()


@cl.action_callback("how_to_use")
async def handle_how_to_use(action: cl.Action):
    await cl.Message(
        content=""" How to use CosmosPumps AI:

1. Add a Website
Click * Add Website* → Enter a URL → I'll crawl and learn from it

2. Upload a Document
Click the  button → Select PDF, DOCX, or TXT → Ask questions about it

3. Upload an Image
Click the  button → Select an image → I'll extract and analyze its content

4. Ask Questions
Just type your question! I'll search all your uploaded sources and answer accurately.

 Tips:
- You can upload multiple documents in the same session
- Uploaded content is prioritized over web data in answers
- Sources are always cited in responses
- Session data is cleared when you start a new chat"""
    ).send()



@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id", "default")

    if not message.elements and is_url(message.content):
        url = message.content.strip()
        status_msg  = await cl.Message(content=f" Fetching `{url}`...").send()
        try:

            async with httpx.AsyncClient(
                    timeout=60,
                    follow_redirects=True,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                    }
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html_content = resp.text

            print(f"[FETCH] Status: {resp.status_code}, Size: {len(html_content)} chars")

            await status_msg.stream_token("  Fetched!")

            process_msg = await cl.Message(content="⚙️ Processing content...").send()

            def parse_and_convert():
                soup = BeautifulSoup(html_content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                md = markdownify.markdownify(str(soup), heading_style="ATX")
                return md

            markdown_content = await asyncio.to_thread(parse_and_convert)
            if len(markdown_content) > 100_000:
                markdown_content = markdown_content[:100_000]
            print(f"[MARKDOWN] Size: {len(markdown_content)} chars")

            await process_msg.stream_token("  Converted!")

            chunk_msg = await cl.Message(content=" Chunking and embedding...").send()


            chunks = await da.process_webhook_data(
                markdown_content=markdown_content,
                html_content=html_content,
                url=url,
                image_scan=True,
                use_llm_tables=True
            )



            print("[DEBUG] Stream token sent")

            async def progress(batch_num, total):
                await chunk_msg.stream_token(f" {batch_num}/{total}")

            retriever = await da.update_and_get_retriever(
                chunks, url, thread_id,
                progress_callback=progress
            )
            webHookListner.set_retriever_for_session(thread_id, retriever)

            await chunk_msg.stream_token("  Done! Ask me anything!")
            await process_msg.update()

        except Exception as e:
            await cl.Message(content=f" Failed to fetch `{url}`: {e}").send()

        return

    if message.elements:

        for element in message.elements:
            if "image" in element.mime:

                with open(element.path, "rb") as f:
                    image_bytes = f.read()

                extracted,retriever = await da.update_retriever_from_image_bytes(image_bytes,session_id=thread_id)

                webHookListner.set_retriever_for_session(thread_id, retriever)

                cl.user_session.set("last_upload_content", extracted)
                cl.user_session.set("last_upload_name", element.name or "uploaded image")

                await cl.Message(
                    content=f" Image added to knowledge"
                ).send()


            elif element.mime in [
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain"
            ]:
                with open(element.path, "rb") as f:
                    doc_bytes = f.read()

                retriever = await da.update_retriever_from_document_bytes(
                    doc_bytes,
                    mime_type=element.mime,
                    filename=element.name,
                    session_id=thread_id
                )

                webHookListner.set_retriever_for_session(thread_id, retriever)

                if "pdf" in element.mime:
                    extracted = da.extract_text_from_pdf(doc_bytes)
                elif "wordprocessingml" in element.mime:
                    extracted = da.extract_text_from_docx(doc_bytes)
                else:
                    extracted = doc_bytes.decode("utf-8", errors="ignore")
                cl.user_session.set("last_upload_content", extracted)
                cl.user_session.set("last_upload_name", element.name)

                await cl.Message(content=f" Document **{element.name}** added to knowledge.").send()

            else:
                await cl.Message(
                    content=f"️ Unsupported file type: `{element.mime}`. Please upload an image, PDF, DOCX, or TXT."
                ).send()

    last_upload = cl.user_session.get("last_upload_content", "") if message.elements else ""
    last_upload_name = cl.user_session.get("last_upload_name", "") if message.elements else ""
    msg = cl.Message(content="")
    search_step = cl.Step(name="Searching documents...")
    await search_step.send()
    full_answer = ""
    sources = []
    has_started_streaming = False

    async for event in graph.astream_events(
            {
                "question": message.content,
                "session_id":thread_id,
                "last_upload_content": last_upload,
                "last_upload_name": last_upload_name
            },
            config={
                "configurable": {"thread_id": cl.user_session.get("thread_id", "default_user")}
                    },
            version="v2"
    ):


        if event["event"] == "on_chat_model_stream":
            if "final_response" in event.get("tags", []):
                if not has_started_streaming :
                    await search_step.remove()
                    await msg.send()
                    has_started_streaming = True


                content = getattr(event["data"]["chunk"], "content", "")

                if content:
                    await msg.stream_token(content)
                    full_answer += content

        if event["event"] == "on_chain_end" and event["name"] == "retrieve":
            sources = event["data"]["output"].get("sources", [])

    if sources:
        unique_sources = ", ".join(dict.fromkeys(sources))
        source_metadata = f"\n\n*Sources: {unique_sources}*"
        await msg.stream_token(source_metadata)
        full_answer += source_metadata
    await msg.update()

@cl.on_stop
async def on_stop():
    thread_id = cl.user_session.get("thread_id")
    if thread_id:
        webHookListner.clear_session(thread_id)
        print(f"[APP] Session {thread_id} cleaned up")

