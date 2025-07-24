from loader import extract_text_from_source
from splitter import split_text
from retriever import init_vectorstore
from graph_flow import build_graph

if __name__ == "__main__":
    doc_path = r"C:\Users\hp\Documents\Langraph\DOCUNMENT.txt"
    source_text = extract_text_from_source(doc_path)
    chunks = split_text(source_text)

    print(f"✅ Loaded and split {len(chunks)} chunks")

    vectorstore = init_vectorstore(chunks)
    print("📥 Chunks uploaded to vector store.")

    qa_app = build_graph(vectorstore)

    user_question = {"user_question": "How can I sign up?"}
    result = qa_app.invoke(user_question)

    print("\n📄 Customer Support Answer")
    print(f"\n🔹 Question:\n{result['user_question']}")
    print(f"\n🔹 Answer:\n{result['final_answer']}")
