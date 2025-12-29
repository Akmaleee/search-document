import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    MINIO_URL_INTERNAL = os.getenv("MINIO_URL_INTERNAL", "http://localhost:9000")
    MINIO_URL_EXTERNAL = os.getenv("MINIO_URL_EXTERNAL", "http://localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "partnership")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
