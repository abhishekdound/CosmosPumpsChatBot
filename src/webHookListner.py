from fastapi import FastAPI, Request, BackgroundTasks
from dataAcquisition import DataAcquisition
import uvicorn

app = FastAPI()
da = DataAcquisition()

current_retriever = da.update_and_get_retriever([], "init")


@app.post("/webhook/firecrawl")
async def firecrawl_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    if payload.get("type") == "crawl.page":
        data_list = payload.get("data", [])

        for pages in data_list:          # first list
            for page in pages:           # inner list
                markdown = page.get("markdown")
                url = page.get("metadata", {}).get("sourceURL")

                if markdown and url:
                    background_tasks.add_task(sync_data, markdown, url)

    return {"status": "ok"}

def sync_data(markdown, url):
    global current_retriever
    chunks = da.process_webhook_data(markdown, url)
    current_retriever = da.update_and_get_retriever(chunks, url)
    print(f"Chatbot updated with fresh data from {url}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
