import chainlit as cl
from llmHelper import graph



@cl.on_chat_start
async def start():
    cl.user_session.set("memory", [])
    await cl.Message(content="Hello! Ask me anything about the documents.").send()


@cl.on_message
async def main(message: cl.Message):

    memory = cl.user_session.get("memory")

    msg = cl.Message(content="")
    await msg.send()

    result = await cl.make_async(graph.invoke)({
        "question": message.content,
        "chat_history": memory
    })

    answer = result["answer"]


    memory.append(f"User: {message.content}")
    memory.append(f"Assistant: {answer}")
    cl.user_session.set("memory", memory)
    for token in answer.split():
        await msg.stream_token(token + " ")

    await msg.update()