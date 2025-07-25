from typing import TypedDict


from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from emil import send_email

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    recipient: str

def build_graph(db) -> StateGraph:
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    def retrieve(state: GraphState):
        retriever = db.as_retriever()
        docs = retriever.invoke(state["question"])
        context = "\n\n".join([doc.page_content for doc in docs])
        return {
            "question": state["question"],
            "context": context,
            "recipient": state.get("recipient", "")
        }

    def generate(state: GraphState):
        prompt = f"""Answer the question using this context:\n\n{state['context']}\n\nQuestion: {state['question']}"""
        response = llm.invoke(prompt)
        return {
            "question": state["question"],
            "context": state["context"],
            "answer": response.content,
            "recipient": state["recipient"]
        }

    def email_node(state: GraphState):
        subject = f"Response to your query: {state['question'][:50]}"
        body = state["answer"]
        if state.get("recipient"):
            send_email(state["recipient"], subject, body)
        return state

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", RunnableLambda(retrieve))
    graph.add_node("generate", RunnableLambda(generate))
    graph.add_node("send_email", RunnableLambda(email_node))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "send_email")
    graph.add_edge("send_email", END)

    return graph.compile()
