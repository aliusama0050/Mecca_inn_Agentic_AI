# ---------- Google Drive Authentication ----------
from googleapiclient.discovery import build as g_build
from google_auth_oauthlib.flow import InstalledAppFlow as AuthFlow
from google.auth.transport.requests import Request as GRequest

import os, pickle


ACCESS_SCOPE = ['https://www.googleapis.com/auth/drive.readonly']
user_creds = None

if os.path.exists('token.pkl'):
    with open('token.pkl', 'rb') as token_file:
        user_creds = pickle.load(token_file)

if not user_creds or not user_creds.valid:
    if user_creds and user_creds.expired and user_creds.refresh_token:
        user_creds.refresh(GRequest())
    else:
        flow = AuthFlow.from_client_secrets_file(r"a.json", ACCESS_SCOPE)
        user_creds = flow.run_local_server(port=0)
    with open('token.pkl', 'wb') as token_file:
        pickle.dump(user_creds, token_file)

drive_client = g_build('drive', 'v3', credentials=user_creds)

# ---------- List Google Drive Files ----------
drive_query = "mimeType='application/pdf' or mimeType='application/vnd.google-apps.document'"
drive_files = drive_client.files().list(
    q=drive_query,
    pageSize=10,
    fields="files(id, name, mimeType)"
).execute()

available_files = drive_files.get('files', [])
for item in available_files:
    print(f"{item['name']} ({item['id']}) - {item['mimeType']}")



# ---------- LangChain + Qdrant Setup ----------
from dotenv import load_dotenv
from langchain_core.documents import Document as LDoc
from langchain_community.vectorstores import Qdrant as QVec
from langchain.text_splitter import RecursiveCharacterTextSplitter as Splitter
from qdrant_client import QdrantClient as QClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import ChatOpenAI as LLM, OpenAIEmbeddings as Embeddings
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda as RunLambda

load_dotenv()

def read_local_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as docfile:
        return docfile.read()

document_raw_text = read_local_text(r"C:\Users\hp\Documents\Google drive inte\DOCUNMENT.txt")

doc_splitter = Splitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = doc_splitter.split_text(document_raw_text)
doc_objects = [LDoc(page_content=chunk) for chunk in doc_chunks]

embedder = Embeddings()
qclient = QClient(host="localhost", port=6333)

qclient.recreate_collection(
    collection_name="custom_txt_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QVec(
    client=qclient,
    collection_name="custom_txt_collection",
    embeddings=embedder
)

vector_store.add_documents(doc_objects)
print("Documents uploaded to Qdrant collection.")

# ---------- Graph Setup ----------
from typing import TypedDict

class QAState(TypedDict):
    query_text: str
    retrieved_info: str
    output_answer: str

chat_model = LLM(model="gpt-3.5-turbo")

def context_retriever(state: QAState):
    question_text = state["query_text"]
    context_docs = vector_store.as_retriever().invoke(question_text)
    combined_context = "\n\n".join([doc.page_content for doc in context_docs])
    return {"query_text": question_text, "retrieved_info": combined_context}

def answer_generator(state: QAState):
    prompt = f"""Answer the question using the context below:\n\n{state['retrieved_info']}\n\nQuestion: {state['query_text']}"""
    llm_response = chat_model.invoke(prompt)
    return {
        "query_text": state["query_text"],
        "retrieved_info": state["retrieved_info"],
        "output_answer": llm_response.content
    }

workflow = StateGraph(QAState)
workflow.add_node("context_retriever", RunLambda(context_retriever))
workflow.add_node("answer_generator", RunLambda(answer_generator))
workflow.set_entry_point("context_retriever")
workflow.add_edge("context_retriever", "answer_generator")
workflow.add_edge("answer_generator", END)

chat_app = workflow.compile()

# ---------- Run RAG Chat ----------
user_input = {"query_text": "I have an issue setting a different delivery address up"}
response_result = chat_app.invoke(user_input)

print(response_result["output_answer"])
