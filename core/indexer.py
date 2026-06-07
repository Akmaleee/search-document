from elasticsearch import Elasticsearch
from core.file_parser import extract_text
from core.minio_client import upload_to_minio
from core.config import Config
from fastapi import UploadFile
import uuid

# Koneksi
es = Elasticsearch(Config.ELASTICSEARCH_URL)

def index_document(file: UploadFile, doc_id: str = None):
    """Ekstrak teks dan simpan ke ES (Cepat & Tanpa OCR)."""
    # Upload file ke MinIO
    minio_url = upload_to_minio(file, bucket="private")
    
    # Ekstrak teks digital murni dari dokumen (PDF/DOCX)
    file.file.seek(0)
    text = extract_text(file)

    combined_text = text.strip()

    # Buat UUID baru jika doc_id tidak diberikan
    doc_id = doc_id or str(uuid.uuid4())

    # Simpan ke Elasticsearch
    es.index(index="partnership", id=doc_id, document={
        "path": minio_url,
        "content": combined_text
    })

    return {"id": doc_id, "path": minio_url}

# from elasticsearch import Elasticsearch
# from core.file_parser import extract_text, extract_images_with_ocr
# from core.minio_client import upload_to_minio
# from core.config import Config
# from fastapi import UploadFile
# import uuid

# # Koneksi
# es = Elasticsearch(Config.ELASTICSEARCH_URL)

# def index_document(file: UploadFile, doc_id: str = None):
#     """Ekstrak teks, buat embedding, dan simpan ke ES."""
#     # Upload file ke MinIO
#     minio_url = upload_to_minio(file, bucket="private")
    
#     # Ekstrak teks dari dokumen (PDF/DOCX)
#     file.file.seek(0)
#     text = extract_text(file)

#     # 🔹 Ekstrak teks dari gambar dengan OCR
#     file.file.seek(0)
#     ocr_text = extract_images_with_ocr(file)

#     # Gabungkan teks dari dokumen + hasil OCR
#     combined_text = f"{text}\n{ocr_text}".strip()
#     doc_id = doc_id or str(uuid.uuid4())

#     # Simpan ke Elasticsearch
#     es.index(index="partnership", id=doc_id, document={
#         "path": minio_url,
#         "content": combined_text
#     })

#     return {"id": doc_id, "path": minio_url}
