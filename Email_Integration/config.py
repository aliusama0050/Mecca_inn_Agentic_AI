import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "DOCUNMENT.txt"

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
