import asyncio
from core.embeddings import generate_embedding
from core.qdrant_client import store_embeddings
from core.chunking import chunk_text

async def process_document(content: str, filename: str, file_url: str):
    chunks = chunk_text(content)
    print(f"Teks dari {filename} panjangnya {len(content)} karakter")
    print(f"📄 File {filename} dipecah menjadi {len(chunks)} chunk.")

    tasks = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "filename": filename,
            "chunk_index": i,
            "url": file_url,
            "text": chunk
        }
        tasks.append(_embed_and_store(chunk, metadata))
    await asyncio.gather(*tasks)
    print(f"✅ Selesai memproses {filename}.")

async def _embed_and_store(chunk, metadata):
    try:
        vector = generate_embedding(chunk)
        store_embeddings(vector, metadata)
    except Exception as e:
        print(f"❌ Gagal memproses chunk {metadata['chunk_index']}: {e}")
