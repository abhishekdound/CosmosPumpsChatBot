import time

import chainlit as cl
from llmHelper import graph
import webHookListner



from dataAcquisition import DataAcquisition
da=DataAcquisition()


@cl.on_chat_start
async def start():
    thread_id = str(time.time())
    cl.user_session.set("thread_id", thread_id)


@cl.on_message
async def main(message: cl.Message):

    if message.elements:

        for element in message.elements:
            if "image" in element.mime:

                with open(element.path, "rb") as f:
                    image_bytes = f.read()

                retriever = da.update_retriever_from_image_bytes(image_bytes)

                with webHookListner.retriever_lock:
                    webHookListner.current_retriever = retriever

                await cl.Message(
                    content=f" Image added to knowledge"
                ).send()


    msg = cl.Message(content="")
    search_step = cl.Step(name="Searching documents...")
    await search_step.send()
    full_answer = ""
    sources = []
    has_started_streaming = False

    async for event in graph.astream_events(
            {
                "question": message.content
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
    await msg.send()

