import openai
import os

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

embd = OpenAIEmbeddings(client=openai.Embedding())
db = Chroma(persist_directory="./chromadb", embedding_function=embd)
retriever = db.as_retriever()
llm = OpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
ans = qa.run("に関する講義名と講義ID、短い概要を教えてください。")
print(ans)