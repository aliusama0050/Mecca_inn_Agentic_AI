import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict

# Load .env
load_dotenv()

# --- Load & Split Text ---
def load_txt_as_documents(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return raw_text

raw_text = load_txt_as_documents(r"C:\Users\FINE LAPTOP\Desktop\Langchain\langgraph\rag_service.txt")  # replace with your file
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(raw_text)
documents = [Document(page_content=chunk) for chunk in texts]

# --- Qdrant Embeddings Setup ---
embedding_function = OpenAIEmbeddings()
qdrant_client = QdrantClient(host="localhost", port=6333)

qdrant_client.recreate_collection(
    collection_name="rag_txt_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

db = Qdrant(
    client=qdrant_client,
    collection_name="rag_txt_collection",
    embeddings=embedding_function
)

db.add_documents(documents)

# --- LangGraph Setup ---
class GraphState(TypedDict):
    question: str
    context: str
    answer: str

llm = ChatOpenAI(model="gpt-3.5-turbo")

def retrieve(state: GraphState):
    query = state["question"]
    retriever = db.as_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"question": query, "context": context}

def generate(state: GraphState):
    prompt = f"""Answer the question using this context:\n\n{state['context']}\n\nQuestion: {state['question']}"""
    response = llm.invoke(prompt)
    answer = response.content
    return {
        "question": state["question"],
        "context": state["context"],
        "answer": answer
    }

graph = StateGraph(GraphState)
graph.add_node("retrieve", RunnableLambda(retrieve))
graph.add_node("generate", RunnableLambda(generate))
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()
