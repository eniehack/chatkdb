import os
import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# loader = TextLoader().load()
docs = DirectoryLoader("../kdb/", glob="**/*.txt", show_progress=True, use_multithreading=True, loader_cls=TextLoader).load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splited_doc = text_spliter.split_documents(docs)
embd = OpenAIEmbeddings(client=openai.Embedding())
db = Chroma.from_documents(splited_doc, embd, persist_directory="./chromadb")
db.persist()
