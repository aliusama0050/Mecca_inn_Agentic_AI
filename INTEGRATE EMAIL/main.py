from loader import load_txt_as_string
from splitter import split_text_to_chunks
from vectorstore import create_qdrant_vectorstore
from graph_flow import build_graph



if __name__ == "__main__":
    file_path = r"C:\Users\hp\Documents\INTEGRATE EMAIL\DOCUNMENT.txt"

    raw_text = load_txt_as_string(file_path)
    chunks = split_text_to_chunks(raw_text)
    print(f"✅ Loaded and split {len(chunks)} text chunks.")

    db = create_qdrant_vectorstore(chunks)
    print("📤 Uploaded chunks to Qdrant.")

    graph_app = build_graph(db)

    inputs = {
        "question": "how to signup",
        "recipient": "shahzain0141@gmail.com"
    }

    result = graph_app.invoke(inputs)

    print("\n📩 Final Answer:")
    print(result["answer"])
