from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model=os.getenv('GROQ_MODEL'), temperature=0 , streaming=True)