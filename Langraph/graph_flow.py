from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

class QAState(TypedDict):
    user_question: str
    found_context: str
    final_answer: str

def build_graph(vectorstore) -> StateGraph:
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")

    def retrieve_context(state: QAState):
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(state["user_question"])
        context = "\n\n".join([doc.page_content for doc in docs])
        return {
            "user_question": state["user_question"],
            "found_context": context
        }

    def generate_response(state: QAState):
        prompt = f"""Answer the question using this context:\n\n{state['found_context']}\n\nQuestion: {state['user_question']}"""
        result = chat_model.invoke(prompt)
        return {
            "user_question": state["user_question"],
            "found_context": state["found_context"],
            "final_answer": result.content
        }

    graph = StateGraph(QAState)
    graph.add_node("get_context", RunnableLambda(retrieve_context))
    graph.add_node("get_answer", RunnableLambda(generate_response))
    graph.set_entry_point("get_context")
    graph.add_edge("get_context", "get_answer")
    graph.add_edge("get_answer", END)
    
    return graph.compile()
