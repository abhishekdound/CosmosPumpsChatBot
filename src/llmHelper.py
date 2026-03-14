import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dataAcquisition import DataAcquisition

from langchain_classic.retrievers.multi_query import  MultiQueryRetriever



import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")



load_dotenv()

DB_DIR='./chroma_db'


data_acquisition = DataAcquisition()

if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    retriever = data_acquisition.load_vector_DB()
else:
    chunks = data_acquisition.chunks()
    retriever = data_acquisition.save_vector_DB(chunks)
llm = ChatGroq(model=os.getenv('GROQ_MODEL'), temperature=0)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context say:
"The information is not available in the provided source."

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
""")

from typing import TypedDict , List

class State(TypedDict):
    question: str
    context: str
    answer: str
    chat_history: List[str]
    sources: List[str]

async def retrieve(state: State):

    docs = await multi_retriever.ainvoke(state["question"])

    unique_docs = {doc.page_content: doc for doc in docs}.values()

    selected_docs = list(unique_docs)[:4]

    context = "\n\n".join(doc.page_content for doc in selected_docs)

    sources = [doc.metadata.get("source", "Unknown") for doc in selected_docs]

    return {
        "context": context,
        "sources": sources
    }

chain = prompt | llm

async def generate(state: State):

    history = "\n".join(state["chat_history"])

    response = await chain.ainvoke({
        "context": state["context"],
        "question": state["question"],
        "chat_history": history
    })

    return {"answer": response.content}
from langgraph.graph import StateGraph,END

graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.set_entry_point("retrieve")

graph_builder.add_edge("retrieve", "generate")

graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# result = graph.invoke({
#     "question": "where is paris?"
# })
#
# print(result["answer"])


