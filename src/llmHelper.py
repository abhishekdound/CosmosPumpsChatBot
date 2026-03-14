import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic import hub
from langchain.tools import tool
from dataAcquisition import DataAcquisition



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


prompt = ChatPromptTemplate.from_template("""
You are a question answering system.

Use ONLY the information provided in the context.

If the answer is not explicitly mentioned in the context,
respond exactly with:
"The information is not available in the provided source."

Context:
{context}

Question:
{question}
""")

from typing import TypedDict

class State(TypedDict):
    question: str
    context: str
    answer: str

def retrieve(state: State):

    docs = retriever.invoke(state["question"])

    context = "\n\n".join(doc.page_content for doc in docs[:4])

    return {"context": context}

chain = prompt | llm
def generate(state: State):



    response = chain.invoke({
        "context": state["context"],
        "question": state["question"]
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

result = graph.invoke({
    "question": "When did Kohli announce his retirement from Test cricket?"
})

print(result["answer"])


