import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate , ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import src.webHookListner as webHookListner
import asyncio


from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from src.llm import llm

from typing import NotRequired




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
2. If UPLOADED DOCUMENT section exists, answer from it FIRST before anything else.
3. Only use RETRIEVED CONTEXT if the uploaded document doesn't contain the answer.
4. If the context is missing details, use your own internal knowledge to provide a complete answer.
5. If data is structured, format your response using Markdown TABLES.
6. If you don't know and the context doesn't say, simply state you don't know."""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\n{upload_block}\n\nRETRIEVED CONTEXT:\n{context}"),
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
    last_upload_content: NotRequired[str]
    last_upload_name: NotRequired[str]
    session_id:str




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
    token_counter=llm,
    start_on="human",
    include_system=True,
)

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Generate 2 different search queries for the question below.\n"
        "Output only the queries, one per line, no numbering.\n"
        "Question: {question}\n"
        "Queries:"
    )
)

async def retrieve(state: State):
    """Retrieves chunks from all sources (Web/PDF/Image) and reranks them."""
    history = state.get("chat_history", [])
    session_id = state.get("session_id", "default")

    if history:
        history_text = "\n".join([f"{m.type}: {m.content}" for m in history[-3:]])
        standalone_query = await rephrase_chain.ainvoke({
            "question": state["question"],
            "chat_history": history_text
        })
    else:
        standalone_query = state["question"]

    current_retriever = webHookListner.get_retriever_for_session(session_id)

    if current_retriever is None:
        return {
            "context": "No documents available yet.",
            "sources": []
        }
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=current_retriever,
        llm=llm,
        prompt=MULTI_QUERY_PROMPT
    )

    raw_docs = await multi_retriever.ainvoke(standalone_query)

    if not raw_docs:
        return {
            "context": "No relevant documents found.",
            "sources": []
        }


    unique_docs = list({d.metadata.get("chunk_id", d.page_content): d for d in raw_docs}.values())

    reranked_docs = await asyncio.to_thread(reranker.compress_documents, list(unique_docs), standalone_query)

    if not reranked_docs:
        reranked_docs = unique_docs[:5]

    reranked_docs.sort(
        key=lambda d: 0 if (
                "user_image" in d.metadata.get("source", "") or
                f"_{session_id}" in d.metadata.get("source", "")
        ) else 1
    )
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

    trimmed_history = await trimmer.ainvoke(state.get("chat_history", []))
    upload_content = state.get("last_upload_content", "").strip()
    upload_name = state.get("last_upload_name", "Uploaded Document")

    if upload_content:
        truncated = upload_content[:3000]
        upload_block = (
            f"UPLOADED DOCUMENT ({upload_name}) — HIGHEST PRIORITY:\n"
            f'"""\n{truncated}\n"""\n'
            f"Answer from this first before using any other source."
        )
    else:
        upload_block = ""

    response = await chain.ainvoke({
        "context": state["context"],
        "question": state["question"],
        "chat_history": trimmed_history,
        "upload_block": upload_block,
    })

    return {
        "answer": response.content,
        "chat_history": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content)
        ]
    }


from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph,END


# memory = AsyncSqliteSaver.from_conn_string("./checkpoints.db")
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


