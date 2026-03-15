import chainlit as cl
from llmHelper import graph


@cl.on_chat_start
async def start():
    pass


@cl.on_message
async def main(message: cl.Message):
    thread_id = str(cl.user_session.get("id", "default_user"))
    config = {
        "configurable": {"thread_id": thread_id}
    }

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
            config=config,
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

