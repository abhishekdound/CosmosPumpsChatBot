import chainlit as cl
from llmHelper import graph


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello! Ask me anything about the documents."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    config = {
        "configurable": {"thread_id": cl.user_session.get("id")},
        "version": "v2"
    }
    search_step = cl.Step(name="Searching documents...")
    await search_step.send()
    msg = cl.Message(content="")
    full_answer = ""
    sources = []
    has_started_streaming = False

    async for event in graph.astream_events(
            {
                "question": message.content
            },
            config=config
    ):


        if event["event"] == "on_chat_model_stream":
            if "final_response" in event.get("tags", []):
                if not has_started_streaming :
                    await search_step.remove()
                    has_started_streaming = True


                content = getattr(event["data"]["chunk"], "content", "")

                if content:
                    full_answer += content
                    await msg.stream_token(content)

        elif event["event"] == "on_chain_end" and event["name"] == "retrieve":
            sources = event["data"]["output"].get("sources", [])

    if sources:
        unique_sources = ", ".join(dict.fromkeys(sources))
        source_metadata = f"\n\n*Sources: {unique_sources}*"
        await msg.stream_token(source_metadata)
        full_answer += source_metadata
    await msg.send()

