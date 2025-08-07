from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from email_tool import send_email, receive_latest_email

class QAState(TypedDict):
    user_question: str
    found_context: str
    final_answer: str

def build_graph(vectorstore) -> StateGraph:
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")

    def email_input_node(state: dict) -> QAState:
        subject, body, thread_id = receive_latest_email()
        if not body:
            raise ValueError("No new email found.")
        return {
            "user_question": body,
            "found_context": "",
            "final_answer": ""
        }

    def retrieve_context(state: QAState):
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(state["user_question"])
        context = "\n\n".join([doc.page_content for doc in docs])
        return {
            "user_question": state["user_question"],
            "found_context": context,
            "final_answer": ""
        }

    def generate_response(state: QAState):
        prompt = f"""Answer the question using this context:\n\n{state['found_context']}\n\nQuestion: {state['user_question']}"""
        result = chat_model.invoke(prompt)
        return {
            "user_question": state["user_question"],
            "found_context": state["found_context"],
            "final_answer": result.content
        }

    def email_summary_node(state: QAState):
        subject = "Your Chat Summary"
        body = f"""
        Chat Summary:

        Question: {state['user_question']}
        Answer: {state['final_answer']}
        """
        send_email("customer@example.com", subject, body)
        return state

    def notify_unanswered_node(state: QAState):
        if "I'm not sure" in state["final_answer"] or "I don't know" in state["final_answer"]:
            send_email("support@example.com", "Unanswered Query Alert", f"Query: {state['user_question']}\nAnswer: {state['final_answer']}")
        return state

    graph = StateGraph(QAState)
    graph.add_node("email_input", RunnableLambda(email_input_node))
    graph.add_node("get_context", RunnableLambda(retrieve_context))
    graph.add_node("get_answer", RunnableLambda(generate_response))
    graph.add_node("notify_support", RunnableLambda(notify_unanswered_node))
    graph.add_node("send_summary", RunnableLambda(email_summary_node))

    graph.set_entry_point("email_input")
    graph.add_edge("email_input", "get_context")
    graph.add_edge("get_context", "get_answer")
    graph.add_edge("get_answer", "notify_support")
    graph.add_edge("notify_support", "send_summary")
    graph.add_edge("send_summary", END)

    return graph.compile()
