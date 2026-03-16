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







prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict Retrieval-Augmented Generation (RAG) assistant. "
"Use ONLY the provided context to answer the question. "
"If the answer is not in the context, exactly say: "
"'I do not have information about this in my database.' "
"Do NOT use your own external knowledge. "
"Answer in a short complete sentence using the context.\n\n"
"Context:\n{context}"
    )),
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


MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
Generate three alternative search queries that could retrieve
facts, numbers, statistics, or biographical information
related to the question.

Original question:
{question}

Queries:
"""
)


condense_question_system_template = """
You are given a conversation and a follow-up question.

Rewrite the follow-up question so it becomes a standalone question.

Rules:
- Replace pronouns like he, she, they, him, her, it with the correct entity from the conversation.
- Include the entity name explicitly.
- Do NOT answer the question.
- Only return the rewritten question.

Conversation:
{chat_history}

Follow-up question:
{question}

Standalone question:
"""

condense_question_prompt = ChatPromptTemplate.from_messages([
    ("system", condense_question_system_template),
    ("human", "History:\n{chat_history}\n\nQuestion: {question}"),
])
rephrase_chain = condense_question_prompt | llm | StrOutputParser()





reranker_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

reranker = CrossEncoderReranker(
    model=reranker_model,
    top_n=8
)


from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter='approximate',
    start_on="human",
    include_system=True,
)

async def retrieve(state: State):
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=webHookListner.current_retriever,
        prompt=MULTI_QUERY_PROMPT,
        llm=llm
    )
    messages = state.get("chat_history", [])
    trimmed_messages = trimmer.invoke(messages)

    recent_history = ""
    for msg in trimmed_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
        recent_history += f"{role}: {msg.content}\n"

    standalone_question = await rephrase_chain.ainvoke({
        "question": state["question"],
        "chat_history": recent_history
    })

    standalone_question = standalone_question.strip()

    if standalone_question.lower() == state["question"].lower():
        standalone_question = state["question"]

    docs = await multi_retriever.ainvoke(standalone_question)


    seen_ids = set()
    unique_docs = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id")
        if cid not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(cid)
    unique_docs = unique_docs[:25]
    reranked_docs = await asyncio.to_thread(
        reranker.compress_documents,
        unique_docs,
        standalone_question
    )

    selected_docs = reranked_docs[:4]

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in selected_docs
    )

    return {
        "context": context,
        "sources": [doc.metadata.get("source", "Unknown") for doc in selected_docs]
    }

chain = (prompt | llm).with_config({"tags": ["final_response"]})








async def generate(state: State):
    if not state.get("context") or not state["context"].strip():
        answer = "I do not have information about this in my database."
        return {
            "answer": answer,
            "chat_history": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer)
            ]
        }
    messages = state.get("chat_history", [])
    trimmed_messages = trimmer.invoke(messages)



    response = await chain.ainvoke({
        "context": state["context"],
        "question": state["question"],
        "chat_history": trimmed_messages
    })

    return {
        "answer": response.content,
        "chat_history": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content)
        ]
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


