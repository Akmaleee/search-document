from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import io
from fastapi import UploadFile

# ================================
# Model Multimodal CLIP
# ================================
# CLIP (ViT-B/32) mendukung image + text embedding
# Output vector 512 dimensi, cocok dengan Qdrant 512-dim
_model = SentenceTransformer("clip-ViT-B-32")

def generate_image_embedding(file: UploadFile) -> np.ndarray:
    """
    Mengubah file gambar menjadi embedding numerik (512-dim)
    """
    try:
        # Baca file gambar dari UploadFile (FastAPI)
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Generate embedding
        embedding = _model.encode(image, convert_to_numpy=True, normalize_embeddings=True)

        return embedding
    except Exception as e:
        print(f"❌ Gagal membuat embedding gambar: {e}")
        return None

def generate_image_embedding_from_pil(image: Image.Image):
    return _model.encode(image, convert_to_numpy=True, normalize_embeddings=True)