from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.models import VectorParams, Distance
from config import qdrant_client, COLLECTION_NAME
from embedder import get_embedder

def create_qdrant_vectorstore(chunks: list) -> Qdrant:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    documents = [Document(page_content=chunk) for chunk in chunks]
    embedder = get_embedder()
    db = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedder
    )
    db.add_documents(documents)
    return db
