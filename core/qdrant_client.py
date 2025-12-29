from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from core.config import Config
import uuid

# --- buat koneksi Qdrant dengan atau tanpa API key ---
if Config.QDRANT_API_KEY:
    qdrant = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
else:
    qdrant = QdrantClient(url=Config.QDRANT_URL)

# --- buat collection jika belum ada (ganti recreate_collection) ---
collections = [c.name for c in qdrant.get_collections().collections]
if Config.QDRANT_COLLECTION not in collections:
    qdrant.create_collection(
        collection_name=Config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

def store_embeddings(vector, metadata):
    qdrant.upsert(
        collection_name=Config.QDRANT_COLLECTION,
        points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload=metadata)]
    )

def search_similar(query_vector, limit=5):
    results = qdrant.search(Config.QDRANT_COLLECTION, query_vector=query_vector, limit=limit)
    contexts = [r.payload["text"] for r in results]
    return "\n".join(contexts)
