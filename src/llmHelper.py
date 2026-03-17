import os

from dotenv import load_dotenv
from dataAcquisition import DataAcquisition
from langchain_core.prompts import PromptTemplate , ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import webHookListner
import asyncio

from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from llm import llm




os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from transformers.utils import logging
logging.set_verbosity_error()


import warnings
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")



load_dotenv()







SYSTEM_PROMPT = """You are an expert Data Analyst. Your goal is to provide accurate answers.
1. ALWAYS check the provided context first. If the answer is there, use it and cite [Source: name].
2. If the context is missing details, use your own internal knowledge to provide a complete answer.
3. If data is structured, format your response using Markdown TABLES.
4. If you don't know and the context doesn't say, simply state you don't know."""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nCONTEXT:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
from typing import TypedDict , List ,Annotated

class State(TypedDict):
    question: str
    context: str
    answer: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    sources:  List[str]




REPHRASE_TEMPLATE = """Rewrite the follow-up question to be a standalone search query.
Chat History: {chat_history}
Follow-up: {question}
Standalone Query:"""
rephrase_chain = PromptTemplate.from_template(REPHRASE_TEMPLATE) | llm | StrOutputParser()






reranker_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

reranker = CrossEncoderReranker(
    model=reranker_model,
    top_n=5
)


from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter='approximate',
    start_on="human",
    include_system=True,
)
retriever_lock = webHookListner.retriever_lock

async def retrieve(state: State):
    """Retrieves chunks from all sources (Web/PDF/Image) and reranks them."""
    history_text = "\n".join([f"{m.type}: {m.content}" for m in state.get("chat_history", [])[-3:]])
    standalone_query = await rephrase_chain.ainvoke({"question": state["question"], "chat_history": history_text})

    with retriever_lock:
        current_retriever = webHookListner.current_retriever
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=current_retriever,
        llm=llm
    )

    raw_docs = await multi_retriever.ainvoke(standalone_query)





    unique_docs = {d.metadata.get("chunk_id", d.page_content): d for d in raw_docs}.values()

    reranked_docs = await asyncio.to_thread(reranker.compress_documents, list(unique_docs), standalone_query)

    context_parts = []
    source_names = []
    for doc in reranked_docs:
        src = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[SOURCE: {src}]\n{doc.page_content}")
        source_names.append(src)

    return {
        "context": "\n\n".join(context_parts),
        "sources": list(set(source_names))
    }

chain = (qa_prompt | llm).with_config({"tags": ["final_response"]})








async def generate(state: State):
    """Generates response using both context and LLM internal knowledge."""

    trimmed_history = trimmer.invoke(state.get("chat_history", []))

    response = await chain.ainvoke({
        "context": state["context"],
        "question": state["question"],
        "chat_history": trimmed_history
    })

    return {
        "answer": response.content,
        "chat_history": [AIMessage(content=response.content)]
    }


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,END

memory = MemorySaver()

graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.set_entry_point("retrieve")

graph_builder.add_edge("retrieve", "generate")

graph_builder.add_edge("generate", END)

graph = graph_builder.compile(checkpointer=memory)

# result = graph.invoke({
#     "question": "where is paris?"
# })
#
# print(result["answer"])


