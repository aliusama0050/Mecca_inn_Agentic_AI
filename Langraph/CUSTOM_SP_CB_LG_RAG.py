from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict


load_dotenv()
def extract_text_from_source(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as txt:
        return txt.read()
document_path = r"C:\Users\hp\Documents\Langraph\DOCUNMENT.txt"
source_text = extract_text_from_source(document_path)

split_tool = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_parts = split_tool.split_text(source_text)

doc_objects = [Document(page_content=part) for part in text_parts]

embedder = OpenAIEmbeddings()
qdrant_conn = QdrantClient(host="localhost", port=6333)

qdrant_conn.recreate_collection(
    collection_name="DOCUNMENT.txt",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_db = Qdrant(
    client=qdrant_conn,
    collection_name="DOCUNMENT.txt",
    embeddings=embedder
)

vector_db.add_documents(doc_objects)
print("Text chunks uploaded to vector store.")

class QAState(TypedDict):
    user_question: str
    found_context: str
    final_answer: str

chat_model = ChatOpenAI(model="gpt-3.5-turbo")
def retrieve_context(state: QAState):
    user_q = state["user_question"]
    retriever = vector_db.as_retriever()
    retrieved_docs = retriever.invoke(user_q)
    combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"user_question": user_q, "found_context": combined_context}

def generate_response(state: QAState):
    prompt = f"""Answer the question using this context:\n\n{state['found_context']}\n\nQuestion: {state['user_question']}"""
    result = chat_model.invoke(prompt)
    return {
        "user_question": state["user_question"],
        "found_context": state["found_context"],
        "final_answer": result.content
    }

graph_chain = StateGraph(QAState)
graph_chain.add_node("get_context", RunnableLambda(retrieve_context))
graph_chain.add_node("get_answer", RunnableLambda(generate_response))
graph_chain.set_entry_point("get_context")
graph_chain.add_edge("get_context", "get_answer")
graph_chain.add_edge("get_answer", END)
qa_app = graph_chain.compile()

user_input = {"user_question": "How can I sign up"}
response_output = qa_app.invoke(user_input)

print("\n📄 Customer Support Answer")
print("\n🔹 Question:")
print(response_output["user_question"])
print("\n🔹 Answer:")
print(response_output["final_answer"])
