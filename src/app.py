import chainlit as cl
from llmHelper import graph


memory = []

@cl.on_chat_start
async def start():
    global memory
    memory = []
    await cl.Message(content="Hello! Ask me anything about the documents.").send()


@cl.on_message
async def main(message: cl.Message):

    global memory

    question = message.content

    result = graph.invoke({
        "question": question,
        "chat_history": memory
    })

    answer = result["answer"]


    memory.append(f"User: {question}")
    memory.append(f"Assistant: {answer}")

    await cl.Message(content=answer).send()