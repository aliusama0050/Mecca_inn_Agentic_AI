from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.models import VectorParams, Distance
from config import qdrant_client, COLLECTION_NAME
from embedder import get_embedder

def init_vectorstore(docs: list):
    embedder = get_embedder()
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedder
    )
    vectorstore.add_documents([Document(page_content=doc) for doc in docs])
    return vectorstore
