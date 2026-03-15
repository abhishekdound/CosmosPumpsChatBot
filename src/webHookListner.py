from fastapi import FastAPI, Request, BackgroundTasks
from dataAcquisition import DataAcquisition
import uvicorn


import threading
retriever_lock = threading.Lock()

app = FastAPI()
da = DataAcquisition()

current_retriever = da.vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)


@app.post("/webhook/firecrawl")
async def firecrawl_webhook(request: Request, background_tasks: BackgroundTasks):

    payload = await request.json()


    if payload.get("type") == "crawl.page":
        data_list = payload.get("data", [])

        for page in data_list:
            if isinstance(page, dict):
                markdown = page.get("markdown")
                metadata = page.get("metadata", {})
                url = metadata.get("sourceURL") or metadata.get("url")

                if markdown and url:
                    background_tasks.add_task(sync_data, markdown, url)
                    print(f"Queued sync for: {url}")
            else:
                print(f"Unexpected data format: {type(page)}")

    return {"status": "ok"}

def sync_data(markdown, url):
    global current_retriever

    try:
        chunks = da.process_webhook_data(markdown, url)

        with retriever_lock:
            current_retriever = da.update_and_get_retriever(chunks, url)

        print(f"Chatbot updated with fresh data from {url}")

    except Exception as e:
        print(f"Ingestion failed for {url}: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
