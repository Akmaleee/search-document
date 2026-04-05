from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import asyncio
import json

# --- Core Modules ---
from core.minio_client import upload_to_minio
from core.file_parser import extract_text, extract_images_with_ocr, extract_images
from core.worker import process_document
from core.qdrant_client import search_similar
from qdrant_client.models import PointStruct
from core.embeddings import generate_embedding
from core.gemini_chat import generate_answer, extract_metadata_with_gemini # ✅ Import fungsi baru
from core.image_embeddings import generate_image_embedding, generate_image_embedding_from_pil
from core.indexer import index_document
from core.searcher import keyword_search, clean_text
from helpers.prompt import SECTION_PROMPTS
from helpers.response import response

# --- Auth & Database ---
from routers import auth
from core.deps import get_current_user
from prisma import Prisma
from prisma.models import User

# ==========================================
# 1. SETUP LIFESPAN (DATABASE CONNECTION)
# ==========================================
# Global Database Client
db = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Saat aplikasi nyala, connect DB
    print("🔌 Connecting to Database...")
    await db.connect()
    yield
    # Saat aplikasi mati, disconnect DB
    print("🔌 Disconnecting Database...")
    if db.is_connected():
        await db.disconnect()

app = FastAPI(title="AI Service with Gemini + MinIO + Qdrant", lifespan=lifespan)

# Register Router Auth
app.include_router(auth.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. BACKGROUND WORKER (AI ANALYSIS)
# ==========================================
async def run_ai_analysis_and_update(doc_id: str, text_content: str, filename: str, file_url: str):
    """
    Worker yang berjalan di background:
    1. Tanya Gemini: Metadata (Tahun, Tags) + Summary + Sitasi
    2. Update Database Postgres
    3. Lanjut Embedding ke Qdrant
    """
    print(f"🤖 AI Agent sedang bekerja untuk dokumen ID: {doc_id}")
    
    # Kita butuh koneksi client baru untuk thread async ini agar aman
    worker_db = Prisma()
    await worker_db.connect()

    try:
        # A. Ekstrak Metadata & Sitasi via Gemini
        metadata = extract_metadata_with_gemini(text_content)
        
        citations = metadata.get("citations", {})
        
        print(f"✅ AI Result for {filename}: {metadata.get('year')} | Tags: {len(metadata.get('tags', []))}")

        # B. Update Database Postgres
        await worker_db.document.update(
            where={"id": doc_id},
            data={
                "year": metadata.get("year"),
                "tags": metadata.get("tags", []),
                "summary": metadata.get("summary"),
                
                # Simpan Sitasi
                "citation_mla": citations.get("mla"),
                "citation_apa": citations.get("apa"),
                "citation_ieee": citations.get("ieee"),
                "citation_harvard": citations.get("harvard")
            }
        )

        # C. Embedding ke Qdrant (Search Engine)
        await process_document(text_content, filename, file_url)
        
        print(f"🎉 Dokumen {filename} selesai diproses sepenuhnya!")

    except Exception as e:
        print(f"❌ Error di background task: {e}")
    finally:
        if worker_db.is_connected():
            await worker_db.disconnect()


# ==========================================
# 3. ENDPOINT UPLOAD (SMART REPOSITORY)
# ==========================================
@app.post("/upload")
async def upload_file(
    file: UploadFile,
    title: Optional[str] = Form(None), # Opsional, kalau kosong pakai filename
    current_user: User = Depends(get_current_user) # ✅ Wajib Login
):
    try:
        # 1. Upload ke MinIO (Storage Fisik)
        file_url = upload_to_minio(file, bucket="public")

        # 2. Ekstrak Teks & OCR (Dilakukan di awal agar file pointer aman)
        file.file.seek(0)
        text = extract_text(file)
        
        file.file.seek(0)
        ocr_text = extract_images_with_ocr(file)
        
        combined_text = f"{text}\n{ocr_text}".strip()

        # 3. Simpan Data Awal ke Database (Status: Pending AI)
        final_title = title if title else file.filename
        
        # Cek koneksi DB utama (jaga-jaga)
        if not db.is_connected():
            await db.connect()

        new_doc = await db.document.create(
            data={
                "title": final_title,
                "filename": file.filename,
                "file_url": file_url,
                "file_type": file.filename.split('.')[-1],
                "uploaded_by": current_user.id,
                
                # Metadata awal kosong (akan diisi AI nanti)
                "summary": "Sedang diproses oleh AI...",
                "year": None,
                "tags": []
            }
        )

        # 4. Trigger Background Task (Fire & Forget)
        asyncio.create_task(
            run_ai_analysis_and_update(
                doc_id=new_doc.id,
                text_content=combined_text,
                filename=file.filename,
                file_url=file_url
            )
        )

        return response(
            message="Upload berhasil. AI sedang menganalisis dokumen Anda...",
            data={
                "id": new_doc.id,
                "title": final_title,
                "file_url": file_url,
                "status": "processing_background"
            },
            status_code=201
        )

    except Exception as e:
        return response(
            message="Upload gagal",
            data=None,
            error=str(e),
            status_code=500
        )

# ==========================================
# 4. EXISTING ENDPOINTS (CHAT, SEARCH, DLL)
# ==========================================

@app.post("/generate")
async def generate(
    query: str = Form(...),
    section: Optional[str] = Form(None),
    webSearch: Optional[bool] = Form(False)
):
    """
    Chat endpoint gabungan (Qdrant + Gemini).
    """
    try:
        # 1. Generate embedding dari query
        query_vector = generate_embedding(query)

        # 2. Cari konteks dari Qdrant
        context_list = search_similar(query_vector, limit=5)

        # Gabungkan hasil konteks
        if isinstance(context_list, list):
            context_text = "\n\n".join(
                [item.get("payload", {}).get("text", str(item)) for item in context_list]
            )
        else:
            context_text = str(context_list)
            
        clean_context = clean_text(context_text)

        # 3. Tentukan template section
        if section:
            if section not in SECTION_PROMPTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Section '{section}' tidak dikenali."
                )
            section_prompt = SECTION_PROMPTS[section]
        else:
            section_prompt = "Anda adalah asisten AI profesional. Jawab berdasarkan dokumen."

        # 4. Rangkai prompt
        full_prompt = f"""
        {section_prompt}

        === PERTANYAAN ===
        {query}

        === CONTEXT ===
        {clean_context}

        Gunakan format HTML terstruktur (<h3>, <p>, <ul>).
        """

        # 5. Panggil Gemini
        answer = generate_answer(prompt=full_prompt, webSearch=webSearch)
        cleaned_answer = answer.replace("```html", "").replace("```", "").strip()

        return response(
            message="Generate success",
            data={
                "section": section if section else "General",
                "query": query,
                "answer": cleaned_answer,
                "webSearch": webSearch
            },
            status_code=200
        )

    except Exception as e:
        return response(message="Generate failed", error=str(e), status_code=500)
    
@app.post("/chat")
async def chat(
    query: str = Form(...),
    webSearch: Optional[bool] = Form(False)
):
    """Chat umum dengan Gemini + Google Search."""
    try:
        full_prompt = f"""
        Anda adalah asisten AI yang profesional.
        Jawab pertanyaan berikut dengan format HTML terstruktur.
        
        === PERTANYAAN ===
        {query}
        """
        answer = generate_answer(prompt=full_prompt, webSearch=webSearch)
        cleaned_answer = answer.replace("```html", "").replace("```", "").strip()

        wrapped_html = f'<div class="gemini-answer">{cleaned_answer}</div>'

        return response(
            message="Chat success",
            data={"query": query, "answer": wrapped_html},
            status_code=200
        )
    except Exception as e:
        return response(message="Chat failed", error=str(e), status_code=500)
    
@app.post("/index")
async def upload_and_index(file: UploadFile):
    """Index manual ke Elasticsearch (Opsional)."""
    try:
        result = index_document(file)
        return response(message="Indexed success", data=result, status_code=200)
    except Exception as e:
        return response(message="Indexed failed", error=str(e), status_code=500)

@app.get("/search")
def search_documents(q: str, limit: int = 10):
    """Keyword Search (Elasticsearch)."""
    try:
        results = keyword_search(q, size=limit)
        return response(
            message="Search success",
            data={"total_found": len(results), "limit": limit, "results": results},
            status_code=200
        )
    except Exception as e:
        return response(message="Search failed", error=str(e), status_code=500)

@app.post("/chat-document")
async def chat_specific_document(
    query: str = Form(...),
    filename: str = Form(...),
):
    """Chat dengan satu dokumen spesifik."""
    try:
        query_vector = generate_embedding(query)
        context_text = search_similar(query_vector, limit=10, filename=filename)

        if not context_text:
            return response(
                message="Context not found",
                data={"answer": f"Info '{query}' tidak ditemukan di {filename}."},
                status_code=200
            )

        clean_context = clean_text(context_text)
        full_prompt = f"""
        Anda asisten khusus dokumen "{filename}".
        Jawab hanya dari konteks ini:
        {clean_context}
        
        Pertanyaan: {query}
        Format HTML.
        """
        
        answer = generate_answer(prompt=full_prompt, webSearch=False)
        cleaned_answer = answer.replace("```html", "").replace("```", "").strip()

        return response(
            message="Chat success",
            data={"query": query, "filename": filename, "answer": cleaned_answer},
            status_code=200
        )

    except Exception as e:
        return response(message="Chat failed", error=str(e), status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")

# from fastapi import FastAPI, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from core.minio_client import upload_to_minio
# from core.file_parser import extract_text, extract_images_with_ocr, extract_images
# from core.worker import process_document
# from core.qdrant_client import search_similar
# from qdrant_client.models import PointStruct
# from core.embeddings import generate_embedding
# from core.gemini_chat import generate_answer
# from core.image_embeddings import generate_image_embedding, generate_image_embedding_from_pil
# from core.indexer import index_document
# from core.searcher import keyword_search, clean_text
# from helpers.prompt import SECTION_PROMPTS
# from helpers.response import response
# from typing import Optional
# import asyncio
# from routers import auth
# from fastapi import Depends, status
# from core.deps import get_current_user 
# from prisma.models import User
# import json 


# app = FastAPI(title="AI Service with Gemini + MinIO + Qdrant")

# app.include_router(auth.router)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/upload")
# async def upload_file(file: UploadFile):
#     try:
#         # Upload ke MinIO
#         file_url = upload_to_minio(file, bucket="public")

#         # Ekstrak teks dari dokumen (PDF/DOCX)
#         file.file.seek(0)
#         text = extract_text(file)

#         # Ekstrak teks dari gambar dengan OCR
#         file.file.seek(0)
#         ocr_text = extract_images_with_ocr(file)

#         # Gabungkan teks dari dokumen + hasil OCR
#         combined_text = f"{text}\n{ocr_text}".strip()

#         # Proses embedding teks secara async
#         asyncio.create_task(process_document(combined_text, file.filename, file_url))

#         return response(
#             message="Upload file success",
#             data={
#                 "status": "processing",
#                 "file": file.filename,
#                 "url": file_url,
#                 "text_length": len(combined_text)
#             },
#             error=None,
#             status_code=200
#         )

#     except Exception as e:
#         return response(
#             message="Upload file failed",
#             data=None,
#             error=str(e),
#             status_code=500
#         )

# @app.post("/generate")
# async def generate(
#     query: str = Form(...),
#     section: Optional[str] = Form(None),
#     webSearch: Optional[bool] = Form(False)
# ):
#     """
#     Chat endpoint gabungan:
#     - Gunakan query untuk mencari konteks dokumen dari Qdrant.
#     - Dapat mengaktifkan Google Search grounding jika webSearch=True.
#     """

#     try:
#         # 🔹 1. Generate embedding dari query
#         query_vector = generate_embedding(query)

#         # 🔹 2. Cari konteks dari Qdrant
#         context_list = search_similar(query_vector, limit=5)

#         # Gabungkan hasil konteks jadi satu teks
#         if isinstance(context_list, list):
#             context_text = "\n\n".join(
#                 [item.get("payload", {}).get("text", str(item)) for item in context_list]
#             )
#         else:
#             context_text = str(context_list)
            
#         clean_context = clean_text(context_text)

#         # 🔹 3. Tentukan template section (jika ada)
#         if section:
#             if section not in SECTION_PROMPTS:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Section '{section}' tidak dikenali. Pilih salah satu dari: {list(SECTION_PROMPTS.keys())}"
#                 )
#             section_prompt = SECTION_PROMPTS[section]
#         else:
#             section_prompt = "Anda adalah asisten AI Telkomsat. Jawab pertanyaan berikut secara profesional berdasarkan dokumen."

#         # 🔹 4. Rangkai prompt final
#         full_prompt = f"""
#         {section_prompt}

#         === PERTANYAAN ===
#         {query}

#         === CONTEXT ===
#         {clean_context}

#         Gunakan semua konteks di atas untuk menyusun jawaban yang akurat.
#         Gunakan format HTML terstruktur (<h3>, <p>, <ul>, <li>, <strong>).
#         """

#         # 🔹 5. Panggil Gemini (bisa pakai webSearch=True)
#         answer = generate_answer(
#             prompt=full_prompt,
#             webSearch=webSearch
#         )
        
#         cleaned_answer = (
#             answer.replace("```html", "")
#             .replace("```", "")
#             .strip()
#         )

#         # 🔹 6. Kembalikan hasil dengan helper format
#         return response(
#             message="Generate success",
#             data={
#                 "section": section if section else "General",
#                 "query": query,
#                 "answer": cleaned_answer,
#                 "context_used": len(clean_context),
#                 "chunks_used": len(context_list),
#                 "webSearch": webSearch
#             },
#             error=None,
#             status_code=200
#         )

#     except Exception as e:
#         return response(
#             message="Generate failed",
#             data=None,
#             error=str(e),
#             status_code=500
#         )
    
# @app.post("/chat")
# async def chat(
#     query: str = Form(...),
#     webSearch: Optional[bool] = Form(False)
# ):
#     """
#     Chat endpoint dengan Gemini + Google Search (penelusuran web).
#     Mendukung output HTML terstruktur (heading, paragraf, list, dan tabel).
#     """
    
#     try:
#         # 🔹 Prompt lengkap untuk hasil yang bisa tampil rapi di web
#         full_prompt = f"""
#         Anda adalah asisten AI yang profesional dan komunikatif.

#         === PERINTAH FORMAT OUTPUT ===
#         Jawaban Anda HARUS ditulis menggunakan format HTML terstruktur:
#         - Gunakan <h3> untuk judul bagian.
#         - Gunakan <p> untuk paragraf.
#         - Gunakan <ul> dan <li> untuk daftar.
#         - Gunakan <strong> untuk penekanan penting.
#         - Jika diperlukan perbandingan atau data terstruktur, gunakan tabel HTML 

#         === PERTANYAAN ===
#         {query}

#         Jawab dengan format HTML penuh seperti di atas tanpa markdown.
#         """

#         # 🔹 Generate jawaban dari Gemini dengan penelusuran web
#         answer = generate_answer(
#             prompt=full_prompt,
#             webSearch=webSearch
#         )
        
#         # 🧹 Bersihkan backticks & blok kode jika tetap muncul
#         cleaned_answer = (
#             answer.replace("```html", "")
#             .replace("```", "")
#             .strip()
#         )

#         # 🔹 Bungkus output HTML agar mudah ditampilkan di frontend
#         wrapped_html = f"""
#         <div class="gemini-answer">
#             {cleaned_answer}
#         </div>
#         <style>
#         .gemini-answer table {{
#             width: 100%;
#             border-collapse: collapse;
#             margin-top: 12px;
#         }}
#         .gemini-answer th, .gemini-answer td {{
#             border: 1px solid #ccc;
#             padding: 8px;
#             text-align: left;
#         }}
#         .gemini-answer thead {{
#             background-color: #f9f9f9;
#             font-weight: bold;
#         }}
#         .gemini-answer tr:nth-child(even) {{
#             background-color: #f5f5f5;
#         }}
#         </style>
#         """

#         # 🔹 Kembalikan hasil dengan helper format
#         return response(
#             message="Chat success",
#             data={
#                 "query": query,
#                 "answer": wrapped_html,
#             },
#             error=None,
#             status_code=200
#         )

#     except Exception as e:
#         return response(
#             message="Chat failed",
#             data=None,
#             error=str(e),
#             status_code=500
#         )
    
# @app.post("/index")
# async def upload_and_index(file: UploadFile):
#     """Unggah PDF ke MinIO, lalu index ke Elasticsearch."""

#     try:
#         result = index_document(file)
#         return response(
#             message="Indexed success",
#             data=result,
#             error=None,
#             status_code=200
#         )

#     except Exception as e:
#         return response(
#             message="Indexed failed",
#             data=None,
#             error=str(e),
#             status_code=500
#         )

# @app.get("/search")
# def search_documents(q: str, limit: int = 10):
#     """
#     Cari dokumen berdasarkan query dengan batasan jumlah hasil.
#     Contoh request: /search?q=kontrak&limit=20
#     """
    
#     try:
#         # Panggil fungsi search dengan parameter size (limit) yang dinamis
#         results = keyword_search(q, size=limit)
        
#         return response(
#             message="Search success",
#             data={
#                 "total_found": len(results), # Info jumlah dokumen yang ditemukan
#                 "limit_requested": limit,    # Info limit yang diminta
#                 "results": results           # Daftar dokumennya
#             },
#             error=None,
#             status_code=200
#         )

#     except Exception as e:
#         return response(
#             message="Search failed",
#             data=None,
#             error=str(e),
#             status_code=500
#         )

# @app.post("/chat-document")
# async def chat_specific_document(
#     query: str = Form(...),
#     filename: str = Form(...), # 🔹 Parameter wajib: Nama file spesifik
# ):
#     """
#     Chat khusus untuk satu dokumen saja.
#     Membatasi konteks jawaban AI hanya dari file yang dipilih.
#     """
#     try:
#         # 1. Generate embedding dari pertanyaan user
#         query_vector = generate_embedding(query)

#         # 2. Cari konteks TAPI difilter hanya untuk filename tersebut
#         context_text = search_similar(query_vector, limit=10, filename=filename)

#         # Jika tidak ada konteks (misal user bertanya hal yang tidak ada di dokumen)
#         if not context_text:
#             return response(
#                 message="Context not found",
#                 data={
#                     "query": query,
#                     "filename": filename,
#                     "answer": f"Maaf, saya tidak menemukan informasi mengenai '{query}' di dalam dokumen {filename}."
#                 },
#                 status_code=200
#             )

#         clean_context = clean_text(context_text)

#         # 3. Buat Prompt Spesifik
#         full_prompt = f"""
#         Anda adalah asisten AI khusus untuk dokumen: "{filename}".
#         Tugas Anda adalah menjawab pertanyaan user HANYA berdasarkan konteks dokumen yang diberikan di bawah ini.
        
#         === KONTEKS DARI DOKUMEN ===
#         {clean_context}

#         === PERTANYAAN USER ===
#         {query}

#         Instruksi:
#         - Jawab dengan format HTML (<h3>, <p>, <ul>).
#         - Jika jawabannya tidak ada di konteks, katakan dengan jujur bahwa informasi tidak ditemukan di dokumen ini.
#         - Jangan mengarang jawaban di luar konteks yang diberikan.
#         """

#         # 4. Generate jawaban via Gemini (tanpa webSearch agar fokus ke dokumen)
#         answer = generate_answer(prompt=full_prompt, webSearch=False)
        
#         cleaned_answer = answer.replace("```html", "").replace("```", "").strip()

#         return response(
#             message="Chat success",
#             data={
#                 "query": query,
#                 "filename": filename,
#                 "answer": cleaned_answer
#             },
#             status_code=200
#         )

#     except Exception as e:
#         return response(
#             message="Chat failed",
#             data=None,
#             error=str(e),
#             status_code=500
#         )
    
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")
