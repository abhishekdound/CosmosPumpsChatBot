import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain.tools import tool
from dataAcquisition import DataAcquisition


load_dotenv()
DB_DIR='./chroma_db'
@tool
def company_data_search(query: str) -> str:
    """
    The ONLY source for company-specific information, product requirements,
    technical specs, and official updates from the website.
    Use this tool for every query to ensure the answer is grounded in
    the provided web data.
    """
    data_acquisition=DataAcquisition()
    chunks=data_acquisition.chunks()
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("✅ Vector DB found. Loading existing index...")
        retriever= data_acquisition.load_vector_DB()
    else:
        print("❌ No Vector DB found. Running initial sync...")
        retriever=data_acquisition.save_vector_DB(chunks)
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context



llm = ChatGroq(model=os.getenv('GROQ_MODEL'), temperature=0)
prompt = hub.pull("hwchase17/react")
tools=[company_data_search]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor( agent=agent,
                                tools=tools,
                                verbose=True,
                                handle_parsing_errors=True,
                                early_stopping_method="generate",
                                max_iterations=3 )

response = agent_executor.invoke({"input": "When did Kohli announce his retirement from Test cricket?"})
print(response["output"])
