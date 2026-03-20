from fastapi import FastAPI, Request, BackgroundTasks
from dataAcquisition import DataAcquisition
import uvicorn
import asyncio

import threading
retriever_lock = threading.Lock()
da = DataAcquisition()
crawl_retriever = da.build_retriever(session_id=None)

session_retrievers: dict = {}


def get_retriever_for_session(session_id: str):
    with retriever_lock:
        needs_copy = session_id not in session_retrievers
        retriever = session_retrievers.get(session_id, crawl_retriever)

    if needs_copy:
        da.copy_crawl_to_session("pending", session_id)
        retriever = da.build_retriever(session_id)
        with retriever_lock:
            session_retrievers[session_id] = retriever

    return retriever


def set_retriever_for_session(session_id: str, retriever):
    with retriever_lock:
        session_retrievers[session_id] = retriever


def clear_session(session_id: str):
    """Called on chat end — removes session retriever and clears session DB."""
    with retriever_lock:
        session_retrievers.pop(session_id, None)
    da.clear_session(session_id)
    print(f"[SESSION] Cleaned up {session_id}")

app = FastAPI()



@app.post("/webhook/firecrawl")
async def firecrawl_webhook(request: Request, background_tasks: BackgroundTasks):
    """
        Endpoint for Firecrawl webhooks.
        Firecrawl sends a POST request whenever a page crawl is completed.
    """
    payload = await request.json()


    if payload.get("type") == "crawl.page":
        data_list = payload.get("data", [])

        for page in data_list:
            if isinstance(page, dict):
                markdown = page.get("markdown")
                html = page.get("html")
                metadata = page.get("metadata", {})
                url = metadata.get("sourceURL") or metadata.get("url")

                if (markdown or html) and url:
                    background_tasks.add_task(run_sync_data, markdown,html, url)
                    print(f"Queued sync for: {url}")
            else:
                print(f"Unexpected data format: {type(page)}")
    elif payload.get("type") == "crawl.completed":
        print("Crawl completed ")

    return {"status": "ok"}


def run_sync_data(markdown, html, url):
    asyncio.run(sync_data(markdown, html, url))

async def sync_data(markdown,html, url):
    """
        Heavy-lifting function: Chunks markdown, updates ChromaDB, and
        refreshes the Global Ensemble Retriever.
    """

    try:
        chunks =await da.process_webhook_data(markdown, html,url)
        with retriever_lock:
            active_sessions = list(session_retrievers.keys())

        if not active_sessions:
            da.update_and_get_retriever(chunks, url, "pending")
            print(f"No active sessions — stored as pending crawl")
        else:
            for sid in active_sessions:
                retriever = da.update_and_get_retriever(chunks, url, sid)
                with retriever_lock:
                    session_retrievers[sid] = retriever

        print(f"Chatbot updated with fresh data from {url}")

    except Exception as e:
        print(f"Ingestion failed for {url}: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
