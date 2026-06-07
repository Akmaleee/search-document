import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

# Import dari folder core
from core.deps import get_db, get_current_user
from core.models import User, Document # Asumsikan kamu punya model Document di models.py
from core.watermark_detector import check_document_watermark

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str = "Dokumen Tanpa Judul",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint untuk mengupload dokumen kemitraan dan mendeteksi watermark PNJ otomatis.
    """
    # 1. Validasi Ekstensi File
    allowed_extensions = ["pdf", "jpg", "jpeg", "png"]
    file_ext = file.filename.split(".")[-1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Format file tidak didukung. Gunakan PDF, JPG, atau PNG.")

    # 2. Baca file ke memory
    file_bytes = await file.read()
    
    # 3. PROSES DETEKSI WATERMARK MENGGUNAKAN OPENCV
    try:
        is_has_watermark = check_document_watermark(file_bytes, file.filename)
    except Exception as e:
        print(f"Error OpenCV processing: {e}")
        # Jika gagal diproses, default ke False agar sistem tidak crash
        is_has_watermark = False 

    # 4. Simpan Meta Data ke Database
    # (Catatan: Untuk skripsi aslinya, kamu mungkin ingin menyimpan file_bytes ke 
    # Cloud Storage / MinIO lalu menyimpan URL-nya di sini).
    
    new_doc = Document(
        id=str(uuid.uuid4()),
        title=title,
        file_path=file.filename, # Nanti ganti dengan URL cloud storage jika ada
        is_watermarked=is_has_watermark,
        uploader_id=current_user.id
    )
    
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    return {
        "message": "Dokumen berhasil diunggah dan dianalisis.",
        "document_id": new_doc.id,
        "is_pnj_watermark_detected": is_has_watermark
    }