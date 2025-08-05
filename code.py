import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from jira import JIRA


# === Load environment variables ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")


# === Load and process text data ===
def load_txt_as_documents(txt_file: str):
    with open(txt_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)
    return [Document(page_content=chunk) for chunk in chunks]


documents = load_txt_as_documents("DOCUNMENT.txt")


# === Initialize Qdrant Vector DB ===
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(host="localhost", port=6333)

qdrant_client.recreate_collection(
    collection_name="rag_txt_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="rag_txt_collection",
    embeddings=embedding_function
)

vectorstore.add_documents(documents)
print("✅ Documents uploaded to Qdrant.")


# === JIRA Integration ===
def create_jira_ticket(summary: str, description: str):
    try:
        options = {"server": JIRA_URL}
        jira = JIRA(
            options=options,
            basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN)
        )
        issue_dict = {
            'project': {'key': JIRA_PROJECT_KEY},
            'summary': summary,
            'description': description,
            'issuetype': {'name': 'Task'},
        }
        issue = jira.create_issue(fields=issue_dict)
        print(f"🧾 Created Jira issue: {issue.key}")
        return issue.key
    except Exception as e:
        print(f"❌ Jira ticket creation failed: {e}")
        return None


# === LangGraph State Definition ===
class GraphState(TypedDict):
    question: str
    context: str
    answer: str


llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)


# === RAG Retrieval Node ===
def retrieve(state: GraphState):
    query = state["question"]
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"question": query, "context": context}


# === LLM Answer Generation Node ===
def generate(state: GraphState):
    prompt = f"""Answer the question using the context below:\n\n{state['context']}\n\nQuestion: {state['question']}"""
    response = llm.invoke(prompt)
    answer = response.content

    # Trigger Jira if issue-related
    trigger_keywords = ["issue", "problem", "bug", "error", "fail", "help", "support"]
    if any(word in state["question"].lower() for word in trigger_keywords):
        create_jira_ticket(
            summary=f"Support Request: {state['question']}",
            description=f"""Auto-created from RAG system:

Question:
{state['question']}

Answer:
{answer}
"""
        )

    return {
        "question": state["question"],
        "context": state["context"],
        "answer": answer
    }


# === Build LangGraph ===
graph = StateGraph(GraphState)
graph.add_node("retrieve", RunnableLambda(retrieve))
graph.add_node("generate", RunnableLambda(generate))
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()


# === Run Example ===
if __name__ == "__main__":
    user_input = "I have an issue setting a different delivery address up"
    result = app.invoke({"question": user_input})
    print("\n🧠 Answer:\n", result["answer"])
