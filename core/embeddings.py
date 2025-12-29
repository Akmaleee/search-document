# import google.generativeai as genai
# from core.config import Config

# genai.configure(api_key=Config.GOOGLE_API_KEY)

# def generate_embedding(text: str):
#     model = "models/embedding-001"
#     embedding = genai.embed_content(model=model, content=text)
#     return embedding["embedding"]

from sentence_transformers import SentenceTransformer

# _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
_model = SentenceTransformer("clip-ViT-B-32")

def generate_embedding(text: str):
    return _model.encode(text).tolist()