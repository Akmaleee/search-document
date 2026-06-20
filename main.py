from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, status, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, desc, func
import asyncio
import json
from datetime import datetime



# --- Core Modules ---
from core.minio_client import upload_to_minio
from core.file_parser import extract_text, extract_images
from core.worker import process_document
from core.qdrant_client import search_similar
from qdrant_client.models import PointStruct
from core.embeddings import generate_embedding
from core.gemini_chat import generate_answer, extract_metadata_with_gemini
from core.image_embeddings import generate_image_embedding, generate_image_embedding_from_pil
from core.indexer import index_document
from core.searcher import keyword_search, clean_text
from helpers.prompt import SECTION_PROMPTS
from helpers.response import response
from routers import auth

# ✅ Import Mesin Deteksi Watermark OpenCV
from core.watermark_detector import check_document_watermark 

# --- Auth & Database (SQLAlchemy) ---
from core.deps import get_current_user, get_db
from core.database import engine, Base, AsyncSessionLocal
from core.models import User, Document, Favorite, Role, VisibilityStatus, Announcement, Guideline, AnnouncementType
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

class UpdateSummaryRequest(BaseModel):
    summary: str
    
# --- SKEMA REQUEST ADMIN ---
class AdminDocUpdateRequest(BaseModel):
    title: Optional[str] = None
    document_status: Optional[str] = None
    file_status: Optional[str] = None
    is_watermarked: Optional[bool] = None

class AdminUserRoleRequest(BaseModel):
    # ✅ UPDATE: Mendukung Admin, User, dan Dosen
    role: str # "ADMIN", "USER", atau "DOSEN"

class AdminUserStatusRequest(BaseModel):
    active: bool # True (Aktif), False (Banned)
    
class AnnouncementRequest(BaseModel):
    title: str
    content: str
    type: str # "info", "warning", atau "success"
    is_active: Optional[bool] = True

class GuidelineRequest(BaseModel):
    step_number: int
    description: str

class AdminCreateUserRequest(BaseModel):
    full_name: str
    email: str
    nim: str  # Untuk dosen/admin, ini diisi NIP atau NIDN mereka
    prodi: Optional[str] = None
    password: str
    role: str # "ADMIN", "DOSEN", atau "USER"


# ==========================================
# 1. SETUP LIFESPAN (DATABASE CONNECTION)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔌 Connecting to Database & Syncing Tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    print("🔌 Disconnecting Database...")
    await engine.dispose()

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
    print(f"🤖 AI Agent sedang bekerja untuk dokumen ID: {doc_id}")
    
    async with AsyncSessionLocal() as worker_db:
        try:
            metadata = await asyncio.to_thread(extract_metadata_with_gemini, text_content)
            citations = metadata.get("citations", {})
            
            print(f"✅ AI Result for {filename}: {metadata.get('year')} | Tags: {len(metadata.get('tags', []))}")

            result = await worker_db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalars().first()
            
            if doc:
                ai_year = metadata.get("year")
                if ai_year is not None:
                    doc.year = ai_year
                doc.tags = metadata.get("tags", [])
                doc.summary = metadata.get("summary")
                doc.highlight = metadata.get("highlight")
                doc.citation_mla = citations.get("mla")
                doc.citation_apa = citations.get("apa")
                doc.citation_ieee = citations.get("ieee")
                doc.citation_harvard = citations.get("harvard")
                
                doc.ai_status = "COMPLETED"
                
                await worker_db.commit()

            await process_document(text_content, filename, file_url)
            print(f"🎉 Dokumen {filename} selesai diproses sepenuhnya!")

        except Exception as e:
            await worker_db.rollback()
            print(f"❌ Error di background task: {e}")

# ==========================================
# 3. ENDPOINT UPLOAD (SMART REPOSITORY)
# ==========================================
@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    author: str = Form(...),
    author_degree: str = Form(...),
    publication_date: str = Form(...),
    language: str = Form(...),
    tags_input: str = Form(...),
    category: str = Form(...),
    prodi: str = Form(...),
    document_status: str = Form(...),
    file_status: str = Form(...),
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db) 
):
    file_bytes = await file.read()
    try:
        is_has_watermark = check_document_watermark(file_bytes, file.filename)
        print(f"💧 Hasil Deteksi Watermark PNJ: {is_has_watermark}")
    except Exception as e:
        print(f"⚠️ Error OpenCV processing: {e}")
        is_has_watermark = False 
        
    file.file.seek(0)

    if not is_has_watermark:
        return response(
            message="Upload ditolak. Watermark kemitraan resmi Politeknik Negeri Jakarta (PNJ) tidak terdeteksi pada dokumen. Silakan unggah dokumen yang valid.",
            data=None,
            status_code=400 
        )

    try:
        waktu_sekarang = int(datetime.now().timestamp())
        file.filename = f"{waktu_sekarang}_{file.filename}"

        file_url = upload_to_minio(file, bucket="public")

        file.file.seek(0)
        text = extract_text(file)
        
        combined_text = text.strip()
        
        if not combined_text:
             return response(
                message="Dokumen ditolak. Sistem membutuhkan dokumen PDF berbasis teks, bukan hasil scan gambar.",
                data=None,
                status_code=400
            )

        parsed_date = datetime.strptime(publication_date, "%Y-%m-%d").date()
        doc_year = parsed_date.year 
        
        tags_list = [tag.strip() for tag in tags_input.split(",")] if tags_input else []

        new_doc = Document(
            title=title,
            author=author,
            author_degree=author_degree,
            publication_date=parsed_date,
            language=language,
            category=category,
            prodi=prodi,
            document_status=document_status,
            file_status=file_status,
            summary="Sedang diproses oleh AI...",
            highlight="Sedang diproses oleh AI...",
            tags=tags_list,
            filename=file.filename,
            file_url=file_url,
            file_type=file.filename.split('.')[-1],
            file_size_bytes=len(file_bytes),
            uploaded_by=current_user.id,
            year=doc_year,
            is_watermarked=True
        )
        
        db.add(new_doc)
        await db.commit()
        await db.refresh(new_doc)

        background_tasks.add_task(
            run_ai_analysis_and_update,
            doc_id=new_doc.id,
            text_content=combined_text,
            filename=file.filename,
            file_url=file_url
        )

        return response(
            message="Upload berhasil. Dokumen valid dan AI sedang menganalisisnya...",
            data={
                "id": new_doc.id,
                "title": title,
                "file_url": file_url,
                "is_pnj_watermarked": True,
                "status": "processing_background"
            },
            status_code=201
        )

    except Exception as e:
        return response(
            message="Upload gagal karena kesalahan server",
            data=None,
            error=str(e),
            status_code=500
        )

# ==========================================
# 4. EXISTING ENDPOINTS (CHAT, SEARCH, DLL)
# ==========================================

@app.get("/documents")
async def get_all_documents(
    search: Optional[str] = None,
    year: Optional[int] = None,       
    category: Optional[str] = None,   
    limit: int = 12, 
    skip: int = 0,
    db: AsyncSession = Depends(get_db)
    # ✅ UPDATE: Menghapus "current_user: User = Depends(get_current_user)" agar GUEST bisa mengakses Endpoint ini
):
    try:
        stmt = select(Document).where(Document.document_status == VisibilityStatus.PUBLIK)
        es_highlights_map = {}

        if search:
            try:
                es_results = keyword_search(search, size=50)
                es_ids = []
                
                for res in es_results:
                    es_ids.append(res["id"])
                    if res.get("highlight") and len(res["highlight"]) > 0:
                        es_highlights_map[res["id"]] = res["highlight"][0]["sentence"]
                
                if es_ids:
                    stmt = stmt.where(
                        or_(
                            Document.id.in_(es_ids),
                            Document.title.ilike(f"%{search}%"),
                            Document.author.ilike(f"%{search}%")
                        )
                    )
                else:
                    stmt = stmt.where(
                        or_(
                            Document.title.ilike(f"%{search}%"),
                            Document.author.ilike(f"%{search}%")
                        )
                    )
            except Exception as e:
                print(f"⚠️ Elasticsearch gagal/mati, fallback ke SQL murni: {e}")
                stmt = stmt.where(
                    or_(
                        Document.title.ilike(f"%{search}%"),
                        Document.author.ilike(f"%{search}%"),
                        Document.highlight.ilike(f"%{search}%")
                    )
                )
                
        if year:
            stmt = stmt.where(Document.year == year) 
            
        if category and category not in ["Semua Kategori", ""]:
            stmt = stmt.where(Document.category == category)

        total_stmt = select(func.count()).select_from(stmt.subquery())
        total_count = await db.scalar(total_stmt)

        stmt = stmt.order_by(desc(Document.created_at)).offset(skip).limit(limit)
        result = await db.execute(stmt)
        documents = result.scalars().all()

        data = []
        for doc in documents:
            prodi_name = "Umum"
            if doc.prodi:
                 prodi_name = doc.prodi.value if hasattr(doc.prodi, 'value') else str(doc.prodi)
                 
            smart_snippet = es_highlights_map.get(doc.id, None)
                 
            data.append({
                "id": doc.id,
                "title": doc.title,
                "author": doc.author,
                "prodi": prodi_name,
                "category": doc.category, 
                "year": doc.year,
                "highlight": doc.highlight,
                "es_highlight": smart_snippet,
                "tags": doc.tags or [],
                "views_count": doc.views_count or 0,
                "downloads_count": doc.downloads_count or 0,
                "file_url": doc.file_url,
                "ai_status": doc.ai_status.value if hasattr(doc.ai_status, 'value') else str(doc.ai_status),
                "is_watermarked": doc.is_watermarked
            })

        return response(
            message="Berhasil mengambil data dokumen",
            data={
                "total": total_count,
                "documents": data
            },
            status_code=200
        )

    except Exception as e:
        return response(
            message="Gagal mengambil data dokumen",
            data=None,
            error=str(e),
            status_code=500
        )
        
@app.get("/documents/me")
async def get_my_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        stmt = select(Document).where(Document.uploaded_by == current_user.id).order_by(desc(Document.created_at))
        result = await db.execute(stmt)
        documents = result.scalars().all()

        data = []
        for doc in documents:
            prodi_name = "Umum"
            if doc.prodi:
                prodi_name = doc.prodi.value if hasattr(doc.prodi, 'value') else str(doc.prodi)
            
            data.append({
                "id": doc.id,
                "title": doc.title,
                "author": doc.author,
                "prodi": prodi_name,
                "year": doc.year,
                "highlight": doc.highlight,
                "tags": doc.tags or [],
                "views_count": doc.views_count or 0,
                "downloads_count": doc.downloads_count or 0,
                "file_url": doc.file_url,
                "ai_status": doc.ai_status.value if hasattr(doc.ai_status, 'value') else str(doc.ai_status),
                "is_watermarked": doc.is_watermarked,
                "created_at": doc.created_at.isoformat() if doc.created_at else None
            })

        return response(
            message="Berhasil mengambil daftar dokumen pribadi Anda",
            data={
                "total": len(data),
                "documents": data
            },
            status_code=200
        )

    except Exception as e:
        return response(
            message="Gagal mengambil daftar dokumen pribadi",
            data=None,
            error=str(e),
            status_code=500
        )        
        
@app.patch("/documents/{doc_id}/summary")
async def update_document_summary(
    doc_id: str,
    request: UpdateSummaryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        stmt = select(Document).where(Document.id == doc_id)
        result = await db.execute(stmt)
        doc = result.scalars().first()

        if not doc:
            return response(message="Dokumen tidak ditemukan", data=None, status_code=404)

        if doc.uploaded_by != current_user.id:
            return response(
                message="Akses ditolak. Anda hanya dapat mengedit dokumen yang Anda unggah sendiri.",
                data=None,
                status_code=403
            )

        doc.summary = request.summary
        await db.commit()
        await db.refresh(doc)

        return response(
            message="Ringkasan dokumen berhasil diperbarui",
            data={"id": doc.id, "new_summary": doc.summary},
            status_code=200
        )

    except Exception as e:
        await db.rollback() 
        return response(message="Gagal memperbarui ringkasan dokumen", data=None, error=str(e), status_code=500)
        
# ==========================================
# 👑 FITUR PANEL ADMIN (DASHBOARD & MODERASI)
# ==========================================

def verify_admin(user: User):
    if user.role != Role.ADMIN:
        raise Exception("Akses ditolak. Area ini khusus Administrator.")

# --- 1. DASHBOARD STATISTIK ---
@app.get("/admin/dashboard")
async def get_admin_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        verify_admin(current_user)

        total_docs = await db.scalar(select(func.count(Document.id)))
        total_users = await db.scalar(select(func.count(User.id)))
        total_publik = await db.scalar(select(func.count(Document.id)).where(Document.document_status == VisibilityStatus.PUBLIK))
        total_private = await db.scalar(select(func.count(Document.id)).where(Document.document_status == VisibilityStatus.PRIVATE))

        top_views_query = await db.execute(select(Document).order_by(desc(Document.views_count)).limit(5))
        top_docs = top_views_query.scalars().all()
        
        top_docs_data = [
            {"id": d.id, "title": d.title, "views_count": d.views_count, "author": d.author} 
            for d in top_docs
        ]

        return response(
            message="Data dashboard admin berhasil diambil",
            data={
                "metrics": {
                    "total_documents": total_docs or 0,
                    "total_users": total_users or 0,
                    "public_documents": total_publik or 0,
                    "private_documents": total_private or 0
                },
                "top_documents": top_docs_data
            },
            status_code=200
        )
    except Exception as e:
        return response(message="Gagal memuat dashboard", error=str(e), status_code=403)


# --- 2. MODERASI DOKUMEN ---
@app.patch("/admin/documents/{doc_id}")
async def admin_moderate_document(
    doc_id: str,
    request: AdminDocUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        verify_admin(current_user)
        
        doc_query = await db.execute(select(Document).where(Document.id == doc_id))
        doc = doc_query.scalars().first()
        
        if not doc:
            return response(message="Dokumen tidak ditemukan", status_code=404)

        if request.title is not None:
            doc.title = request.title
            
        if request.document_status is not None:
            doc.document_status = VisibilityStatus(request.document_status)
            
        if request.file_status is not None:
            doc.file_status = VisibilityStatus(request.file_status)
            
        if request.is_watermarked is not None:
            doc.is_watermarked = request.is_watermarked 

        await db.commit()
        await db.refresh(doc)

        return response(message="Dokumen berhasil dimoderasi", data={"id": doc.id}, status_code=200)
    except Exception as e:
        await db.rollback()
        return response(message="Gagal memoderasi dokumen", error=str(e), status_code=500)

@app.delete("/admin/documents/{doc_id}")
async def admin_delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        verify_admin(current_user)
        
        doc_query = await db.execute(select(Document).where(Document.id == doc_id))
        doc = doc_query.scalars().first()
        
        if not doc:
            return response(message="Dokumen tidak ditemukan", status_code=404)

        await db.delete(doc)
        await db.commit()
        
        return response(message="Dokumen berhasil dihapus dari sistem", data=None, status_code=200)
    except Exception as e:
        await db.rollback()
        return response(message="Gagal menghapus dokumen", error=str(e), status_code=500)


# --- 3. MANAJEMEN PENGGUNA ---
@app.get("/admin/users")
async def admin_get_all_users(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        verify_admin(current_user)
        
        users_query = await db.execute(select(User).order_by(desc(User.created_at)))
        users = users_query.scalars().all()
        
        user_data = []
        for u in users:
            user_data.append({
                "id": u.id,
                "full_name": u.full_name,
                "email": u.email,
                "nim": u.nim,
                "role": u.role.value if hasattr(u.role, 'value') else str(u.role),
                "active": u.active
            })
            
        return response(message="Daftar pengguna berhasil diambil", data=user_data, status_code=200)
    except Exception as e:
        return response(message="Gagal mengambil daftar pengguna", error=str(e), status_code=500)

@app.post("/admin/users")
async def admin_create_user(
    request: AdminCreateUserRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Admin membuat akun baru secara langsung (Bypass verifikasi email)."""
    import re
    import secrets # 👈 Tambahan untuk membuat dummy token
    
    # Pastikan fungsi hash ini bisa diimport. Jika error, sesuaikan path-nya (misal dari core.security)
    from routers.auth import get_password_hash 
    
    try:
        verify_admin(current_user)
        
        # 1. GATEKEEPER: Validasi Email
        if not request.email.endswith(".pnj.ac.id"):
            return response(message="Gagal membuat akun. Wajib menggunakan email resmi PNJ (.pnj.ac.id)", status_code=400)

        # 2. GATEKEEPER: Validasi NIM/NIP
        if not request.nim.isdigit() or len(request.nim) > 10:
            return response(message="Gagal membuat akun. NIM/NIP harus berupa angka dan maksimal 10 digit.", status_code=400)

        # 3. GATEKEEPER: Validasi Password
        if len(request.password) < 8:
            return response(message="Gagal membuat akun. Password minimal 8 karakter.", status_code=400)
        if not re.search(r"[A-Z]", request.password):
            return response(message="Gagal membuat akun. Password harus mengandung minimal 1 huruf besar.", status_code=400)
        if not re.search(r"[a-z]", request.password):
            return response(message="Gagal membuat akun. Password harus mengandung minimal 1 huruf kecil.", status_code=400)
        if not re.search(r"\d", request.password):
            return response(message="Gagal membuat akun. Password harus mengandung minimal 1 angka.", status_code=400)
        if not re.search(r"[@$!%*?&#]", request.password):
            return response(message="Gagal membuat akun. Password harus mengandung minimal 1 simbol khusus (@$!%*?&#).", status_code=400)

        # 4. Cek Database
        email_check = await db.execute(select(User).where(User.email == request.email))
        if email_check.scalars().first():
            return response(message="Gagal membuat akun. Email tersebut sudah terdaftar.", status_code=400)
            
        nim_check = await db.execute(select(User).where(User.nim == request.nim))
        if nim_check.scalars().first():
            return response(message="Gagal membuat akun. NIM/NIP tersebut sudah terdaftar.", status_code=400)

        # 5. BERSIHKAN DATA PRODI (Ubah string kosong "" jadi None agar DB Enum tidak error)
        safe_prodi = request.prodi if request.prodi and request.prodi.strip() != "" else None

        # 6. Hash Password & Buat Dummy Token
        hashed_pwd = get_password_hash(request.password)
        dummy_token = secrets.token_urlsafe(32) # 👈 Buat token acak agar lolos validasi NOT NULL database

        # 7. Simpan Akun Baru
        new_user = User(
            email=request.email,
            password_hash=hashed_pwd,
            full_name=request.full_name,
            nim=request.nim,
            prodi=safe_prodi, # 👈 Gunakan prodi yang sudah dibersihkan
            role=Role(request.role), 
            active=True,
            is_verified=True, 
            verification_token=dummy_token # 👈 Masukkan dummy token
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        return response(
            message=f"Akun {request.role} atas nama {request.full_name} berhasil dibuat.",
            data={"id": new_user.id, "email": new_user.email},
            status_code=201
        )
    except Exception as e:
        await db.rollback()
        import traceback
        traceback.print_exc() 
        return response(message="Gagal membuat akun", error=str(e), status_code=500)
    
@app.patch("/admin/users/{user_id}/role")
async def admin_change_user_role(
    user_id: str,
    request: AdminUserRoleRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        verify_admin(current_user)
        
        if user_id == current_user.id:
            return response(message="Anda tidak dapat mengubah role Anda sendiri", status_code=400)
            
        user_query = await db.execute(select(User).where(User.id == user_id))
        user = user_query.scalars().first()
        
        user.role = Role(request.role)
        await db.commit()
        
        return response(message=f"Jabatan pengguna diubah menjadi {request.role}", status_code=200)
    except Exception as e:
        await db.rollback()
        return response(message="Gagal mengubah jabatan", error=str(e), status_code=500)

@app.patch("/admin/users/{user_id}/status")
async def admin_toggle_user_status(
    user_id: str,
    request: AdminUserStatusRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        verify_admin(current_user)
        
        if user_id == current_user.id:
            return response(message="Anda tidak dapat memblokir akun Anda sendiri", status_code=400)
            
        user_query = await db.execute(select(User).where(User.id == user_id))
        user = user_query.scalars().first()
        
        user.active = request.active
        await db.commit()
        
        status_msg = "diaktifkan" if request.active else "diblokir"
        return response(message=f"Akun pengguna berhasil {status_msg}", status_code=200)
    except Exception as e:
        await db.rollback()
        return response(message="Gagal mengubah status akun", error=str(e), status_code=500)        
        
@app.get("/announcements")
async def get_active_announcements(db: AsyncSession = Depends(get_db)):
    query = await db.execute(
        select(Announcement)
        .where(Announcement.is_active == True)
        .order_by(desc(Announcement.created_at))
    )
    
    announcements = query.scalars().all()
    
    announcements_data = [
        {
            "id": ann.id,
            "title": ann.title,
            "content": ann.content,
            "type": ann.type.value if hasattr(ann.type, "value") else ann.type,
            "is_active": ann.is_active,
            "created_by": ann.created_by,
            "created_at": ann.created_at.isoformat() if ann.created_at else None,
            "updated_at": ann.updated_at.isoformat() if ann.updated_at else None
        }
        for ann in announcements
    ]

    return response(message="Berhasil", data=announcements_data, status_code=200)

@app.get("/admin/announcements")
async def admin_get_all_announcements(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    verify_admin(current_user)
    
    query = await db.execute(select(Announcement).order_by(desc(Announcement.created_at)))
    announcements = query.scalars().all()
    
    announcements_data = [
        {
            "id": ann.id,
            "title": ann.title,
            "content": ann.content,
            "type": ann.type.value if hasattr(ann.type, "value") else ann.type,
            "is_active": ann.is_active,
            "created_by": ann.created_by,
            "created_at": ann.created_at.isoformat() if ann.created_at else None,
            "updated_at": ann.updated_at.isoformat() if ann.updated_at else None
        }
        for ann in announcements
    ]

    return response(message="Berhasil mengambil semua pengumuman", data=announcements_data, status_code=200)

@app.post("/admin/announcements")
async def create_announcement(
    req: AnnouncementRequest, 
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    verify_admin(current_user)
    new_ann = Announcement(
        title=req.title, content=req.content, 
        type=AnnouncementType(req.type), is_active=req.is_active,
        created_by=current_user.id
    )
    db.add(new_ann)
    await db.commit()
    return response(message="Pengumuman ditambahkan", data={"id": new_ann.id}, status_code=201)

@app.patch("/admin/announcements/{ann_id}")
async def update_announcement(
    ann_id: str, 
    req: AnnouncementRequest, 
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    verify_admin(current_user)
    query = await db.execute(select(Announcement).where(Announcement.id == ann_id))
    ann = query.scalars().first()
    if not ann:
        return response(message="Tidak ditemukan", status_code=404)
    
    ann.title = req.title
    ann.content = req.content
    ann.type = AnnouncementType(req.type)
    ann.is_active = req.is_active
    await db.commit()
    return response(message="Pengumuman diperbarui", status_code=200)

# ==========================================
# 📝 API PANDUAN UPLOAD (GUIDELINES)
# ==========================================
@app.get("/guidelines")
async def get_guidelines(db: AsyncSession = Depends(get_db)):
    query = await db.execute(select(Guideline).order_by(Guideline.step_number))
    guidelines = query.scalars().all()
    
    guidelines_data = [
        {
            "id": guide.id,
            "step_number": guide.step_number,
            "description": guide.description,
            "created_at": guide.created_at.isoformat() if guide.created_at else None,
            "updated_at": guide.updated_at.isoformat() if guide.updated_at else None
        }
        for guide in guidelines
    ]

    return response(message="Berhasil", data=guidelines_data, status_code=200)

@app.post("/admin/guidelines")
async def create_guideline(
    req: GuidelineRequest, 
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    verify_admin(current_user)
    new_guide = Guideline(step_number=req.step_number, description=req.description)
    db.add(new_guide)
    await db.commit()
    return response(message="Panduan ditambahkan", status_code=201)

@app.patch("/admin/guidelines/{guide_id}")
async def update_guideline(
    guide_id: str, 
    req: GuidelineRequest, 
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    verify_admin(current_user)
    query = await db.execute(select(Guideline).where(Guideline.id == guide_id))
    guide = query.scalars().first()
    if not guide:
        return response(message="Tidak ditemukan", status_code=404)
    
    guide.step_number = req.step_number
    guide.description = req.description
    await db.commit()
    return response(message="Panduan diperbarui", status_code=200)

# ==========================================
# 📊 API STATISTIK PUBLIK (HOME PAGE)
# ==========================================
@app.get("/statistics")
async def get_public_statistics(db: AsyncSession = Depends(get_db)):
    try:
        now = datetime.now()
        
        total_docs = await db.scalar(select(func.count(Document.id)))
        this_month_count = await db.scalar(
            select(func.count(Document.id))
            .where(func.extract('year', Document.created_at) == now.year)
            .where(func.extract('month', Document.created_at) == now.month)
        )
        
        year_stats_query = await db.execute(
            select(
                func.extract('year', Document.created_at).label('year'),
                func.count(Document.id).label('count')
            )
            .group_by(func.extract('year', Document.created_at))
            .order_by(desc(func.extract('year', Document.created_at)))
        )
        
        docs_per_year = []
        for row in year_stats_query.all():
            if row.year: 
                docs_per_year.append({
                    "year": int(row.year),
                    "count": row.count
                })
                
        return response(
            message="Berhasil mengambil statistik",
            data={
                "total_documents": total_docs or 0,
                "this_month_uploads": this_month_count or 0,
                "documents_per_year": docs_per_year
            },
            status_code=200
        )
    except Exception as e:
        return response(message="Gagal mengambil statistik", error=str(e), status_code=500)


# ==========================================
# 🔹 FITUR FAVORITE (BOOKMARK)
# ==========================================

@app.get("/documents/favorites")
async def get_favorite_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        stmt = (
            select(Document)
            .join(Favorite, Document.id == Favorite.document_id)
            .where(Favorite.user_id == current_user.id)
            .order_by(desc(Favorite.assigned_at))
        )
        
        result = await db.execute(stmt)
        documents = result.scalars().all()

        data = []
        for doc in documents:
            prodi_name = doc.prodi.value if hasattr(doc.prodi, 'value') else str(doc.prodi) if doc.prodi else "Umum"
            
            data.append({
                "id": doc.id,
                "title": doc.title,
                "author": doc.author,
                "prodi": prodi_name,
                "year": doc.year,
                "highlight": doc.highlight,
                "tags": doc.tags or [],
                "views_count": doc.views_count or 0,
                "downloads_count": doc.downloads_count or 0,
                "file_url": doc.file_url,
                "ai_status": doc.ai_status.value if hasattr(doc.ai_status, 'value') else str(doc.ai_status),
                "is_watermarked": doc.is_watermarked,
                "is_favorited": True 
            })

        return response(
            message="Berhasil mengambil dokumen favorit",
            data={"total": len(data), "documents": data},
            status_code=200
        )
    except Exception as e:
        return response(message="Gagal mengambil favorit", error=str(e), status_code=500)


@app.post("/documents/{doc_id}/favorite")
async def toggle_favorite(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        doc_query = await db.execute(select(Document).where(Document.id == doc_id))
        if not doc_query.scalars().first():
            return response(message="Dokumen tidak ditemukan", data=None, status_code=404)

        fav_query = await db.execute(
            select(Favorite).where(
                Favorite.user_id == current_user.id,
                Favorite.document_id == doc_id
            )
        )
        existing_fav = fav_query.scalars().first()

        if existing_fav:
            await db.delete(existing_fav)
            await db.commit()
            return response(
                message="Dokumen dihapus dari favorit", 
                data={"is_favorited": False}, 
                status_code=200
            )
        else:
            new_fav = Favorite(user_id=current_user.id, document_id=doc_id)
            db.add(new_fav)
            await db.commit()
            return response(
                message="Dokumen ditambahkan ke favorit", 
                data={"is_favorited": True}, 
                status_code=201
            )

    except Exception as e:
        await db.rollback()
        return response(message="Gagal mengubah status favorit", error=str(e), status_code=500)

@app.get("/documents/{doc_id}")
async def get_document_detail(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme_optional) # 👈 Kunci Perubahan
):
    """Mengambil detail dokumen. Guest bisa melihat detail, tapi TIDAK bisa download."""
    try:
        stmt = select(Document).where(Document.id == doc_id)
        result = await db.execute(stmt)
        doc = result.scalars().first()

        if not doc:
            return response(message="Dokumen tidak ditemukan", data=None, status_code=404)

        doc.views_count = (doc.views_count or 0) + 1
        await db.commit()
        await db.refresh(doc)

        prodi_name = doc.prodi.value if hasattr(doc.prodi, 'value') else str(doc.prodi) if doc.prodi else "Umum"
        category_name = doc.category.value if hasattr(doc.category, 'value') else str(doc.category) if doc.category else "-"
        doc_status = doc.document_status.value if hasattr(doc.document_status, 'value') else str(doc.document_status) if doc.document_status else "PUBLIK"
        file_stat = doc.file_status.value if hasattr(doc.file_status, 'value') else str(doc.file_status) if doc.file_status else "PUBLIK"

        # --- 1. IDENTIFIKASI USER SECARA SILUMAN (Tanpa 401 Error) ---
        current_user = None
        if token:
            try:
                # Panggil fungsi aslimu untuk mengecek token yang masuk
                current_user = await get_current_user(token=token, db=db)
            except:
                pass # Jika token rusak/kedaluwarsa, biarkan dia menjadi Guest

        # --- 2. LOGIKA DOWNLOAD MAZHAB B SUPER KETAT ---
        actual_file_url = None # DEFAULT: GUEST TIDAK BISA DOWNLOAD SAMA SEKALI

        if current_user:
            # JIKA DIA LOGIN (User/Dosen/Admin)
            if file_stat == "PUBLIK":
                actual_file_url = doc.file_url # Publik boleh di-download user login
            else:
                # JIKA PRIVATE / TERBATAS
                is_author = (current_user.id == doc.uploaded_by)
                is_privileged = current_user.role in [Role.ADMIN, Role.DOSEN]
                
                if is_author or is_privileged:
                    actual_file_url = doc.file_url

        # Format output
        doc_detail = {
            "id": doc.id,
            "title": doc.title,
            "author": doc.author,
            "author_degree": doc.author_degree,
            "publication_date": doc.publication_date.isoformat() if doc.publication_date else None,
            "year": doc.year,
            "language": doc.language,
            "category": category_name,
            "prodi": prodi_name,
            "document_status": doc_status,
            "file_status": file_stat,
            "summary": doc.summary,               
            "highlight": doc.highlight,
            "tags": doc.tags or [],
            "filename": doc.filename,
            "file_url": actual_file_url, # 👈 File URL aman!
            "file_type": doc.file_type,
            "file_size_bytes": doc.file_size_bytes,
            "ai_status": doc.ai_status.value if hasattr(doc.ai_status, 'value') else str(doc.ai_status),
            "is_watermarked": doc.is_watermarked,
            "views_count": doc.views_count,       
            "downloads_count": doc.downloads_count,
            "citations": {                        
                "mla": doc.citation_mla,
                "apa": doc.citation_apa,
                "ieee": doc.citation_ieee,
                "harvard": doc.citation_harvard,
            },
            "uploaded_by": doc.uploaded_by,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }

        return response(
            message="Berhasil mengambil detail dokumen",
            data=doc_detail,
            status_code=200
        )

    except Exception as e:
        return response(message="Gagal mengambil detail dokumen", data=None, error=str(e), status_code=500)
        
# @app.get("/documents/{doc_id}")
# async def get_document_detail(
#     doc_id: str,
#     db: AsyncSession = Depends(get_db),
#     current_user: User = Depends(get_current_user)
#     # Catatan: Kita butuh 'current_user' di sini untuk mengecek apakah dia DOSEN/ADMIN.
#     # Frontend harus me-redirect Guest ke halaman Login jika mereka mencoba membuka halaman Detail.
# ):
#     try:
#         stmt = select(Document).where(Document.id == doc_id)
#         result = await db.execute(stmt)
#         doc = result.scalars().first()

#         if not doc:
#             return response(message="Dokumen tidak ditemukan", data=None, status_code=404)

#         doc.views_count = (doc.views_count or 0) + 1
#         await db.commit()
#         await db.refresh(doc)

#         prodi_name = "Umum"
#         if doc.prodi:
#             prodi_name = doc.prodi.value if hasattr(doc.prodi, 'value') else str(doc.prodi)
            
#         category_name = doc.category.value if hasattr(doc.category, 'value') else str(doc.category) if doc.category else "-"
#         doc_status = doc.document_status.value if hasattr(doc.document_status, 'value') else str(doc.document_status) if doc.document_status else "PUBLIK"
#         file_stat = doc.file_status.value if hasattr(doc.file_status, 'value') else str(doc.file_status) if doc.file_status else "PUBLIK"

#         # ✅ UPDATE RBAC: Pengecekan Izin Download URL File
#         actual_file_url = doc.file_url
        
#         # Jika file statusnya Private / Terbatas, kita harus blokir URL-nya
#         if file_stat in ["PRIVATE", "TERBATAS"]:
#             is_author = (current_user.id == doc.uploaded_by)
#             # ADMIN dan DOSEN memiliki hak istimewa untuk mengunduh semua dokumen
#             is_privileged = current_user.role in [Role.ADMIN, Role.DOSEN]
            
#             # Jika dia bukan penulis aslinya, DAN juga bukan Dosen/Admin, HAPUS URL-nya
#             if not (is_author or is_privileged):
#                 actual_file_url = None

#         doc_detail = {
#             "id": doc.id,
#             "title": doc.title,
#             "author": doc.author,
#             "author_degree": doc.author_degree,
#             "publication_date": doc.publication_date.isoformat() if doc.publication_date else None,
#             "year": doc.year,
#             "language": doc.language,
#             "category": category_name,
#             "prodi": prodi_name,
#             "document_status": doc_status,
#             "file_status": file_stat,
#             "summary": doc.summary,               
#             "highlight": doc.highlight,
#             "tags": doc.tags or [],
#             "filename": doc.filename,
#             "file_url": actual_file_url, # 👈 Menggunakan URL yang sudah melewati filter RBAC
#             "file_type": doc.file_type,
#             "file_size_bytes": doc.file_size_bytes,
#             "ai_status": doc.ai_status.value if hasattr(doc.ai_status, 'value') else str(doc.ai_status),
#             "is_watermarked": doc.is_watermarked,
#             "views_count": doc.views_count,       
#             "downloads_count": doc.downloads_count,
#             "citations": {                        
#                 "mla": doc.citation_mla,
#                 "apa": doc.citation_apa,
#                 "ieee": doc.citation_ieee,
#                 "harvard": doc.citation_harvard,
#             },
#             "uploaded_by": doc.uploaded_by,
#             "created_at": doc.created_at.isoformat() if doc.created_at else None,
#         }

#         return response(
#             message="Berhasil mengambil detail dokumen",
#             data=doc_detail,
#             status_code=200
#         )

#     except Exception as e:
#         return response(message="Gagal mengambil detail dokumen", data=None, error=str(e), status_code=500)


@app.post("/generate")
async def generate(
    query: str = Form(...),
    section: Optional[str] = Form(None),
    webSearch: Optional[bool] = Form(False),
    current_user: User = Depends(get_current_user)
    
):
    try:
        query_vector = generate_embedding(query)
        context_list = search_similar(query_vector, limit=5)

        if isinstance(context_list, list):
            context_text = "\n\n".join(
                [item.get("payload", {}).get("text", str(item)) for item in context_list]
            )
        else:
            context_text = str(context_list)
            
        clean_context = clean_text(context_text)

        if section:
            if section not in SECTION_PROMPTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Section '{section}' tidak dikenali."
                )
            section_prompt = SECTION_PROMPTS[section]
        else:
            section_prompt = "Anda adalah asisten AI profesional. Jawab berdasarkan dokumen."

        full_prompt = f"""
        {section_prompt}

        === PERTANYAAN ===
        {query}

        === CONTEXT ===
        {clean_context}

        Gunakan format HTML terstruktur (<h3>, <p>, <ul>).
        """

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
    webSearch: Optional[bool] = Form(False),
    current_user: User = Depends(get_current_user)
):
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
async def upload_and_index(
    file: UploadFile,
    current_user: User = Depends(get_current_user) 
):
    try:
        result = index_document(file)
        return response(message="Indexed success", data=result, status_code=200)
    except Exception as e:
        return response(message="Indexed failed", error=str(e), status_code=500)


@app.get("/search")
def search_documents(
    q: str, 
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    print(f"\n🔍 Memulai pencarian untuk kata: '{q}'")
    try:
        print("⏳ Memanggil fungsi keyword_search...")
        
        results = keyword_search(q, size=limit)
        
        print(f"✅ Pencarian selesai! Ditemukan {len(results)} hasil.")
        return response(
            message="Search success",
            data={"total_found": len(results), "limit": limit, "results": results},
            status_code=200
        )
    except Exception as e:
        print(f"❌ ERROR saat pencarian: {e}")
        return response(message="Search failed", error=str(e), status_code=500)

    
@app.post("/chat-document")
async def chat_specific_document(
    query: str = Form(...),
    filename: str = Form(...),
    current_user: User = Depends(get_current_user) 
):
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