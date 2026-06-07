import uuid
import enum
from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, ForeignKey, Enum, UniqueConstraint, Date
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

# Pastikan import Base ini sesuai dengan lokasi file database.py Anda
from core.database import Base

# ==========================================
# ENUMS (Daftar Pilihan Tetap untuk Validasi)
# ==========================================
class Role(str, enum.Enum):
    USER = "USER"
    DOSEN = "DOSEN"
    ADMIN = "ADMIN"

class DocumentCategory(str, enum.Enum):
    SKRIPSI = "Skripsi"
    LAPORAN_MAGANG = "Laporan Magang"
    JURNAL = "Jurnal"

class StudyProgram(str, enum.Enum):
    TI = "Teknik Informatika"
    TMJ = "Teknik Multimedia dan Jaringan"
    TMD = "Teknik Multimedia Digital"

class VisibilityStatus(str, enum.Enum):
    PUBLIK = "Publik"
    PRIVATE = "Private"
    TERBATAS = "Terbatas"
    
class AnnouncementType(str, enum.Enum):
    INFO = "info"       # Biru
    WARNING = "warning" # Kuning
    SUCCESS = "success" # Hijau

class AIProcessingStatus(str, enum.Enum):
    PENDING = "PENDING"       # Baru diupload
    PROCESSING = "PROCESSING" # Sedang diekstrak/divalidasi AI
    COMPLETED = "COMPLETED"   # Selesai diringkas & masuk Qdrant
    FAILED = "FAILED"         # Gagal diproses

# ==========================================
# MODELS
# ==========================================

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    
    # Profil Akademik
    nim = Column(String, unique=True, index=True, nullable=True)
    prodi = Column(Enum(StudyProgram), nullable=True)
    
    # ==========================================
    # SISTEM AUTENTIKASI & KEAMANAN (SMTP READY)
    # ==========================================
    is_verified = Column(Boolean, default=False) 
    
    # Untuk Verifikasi Email Saat Register
    verification_token = Column(String, unique=True, index=True, nullable=True)
    
    # Untuk Fitur Lupa Password
    reset_password_token = Column(String, unique=True, index=True, nullable=True)
    reset_password_expires_at = Column(DateTime(timezone=True), nullable=True)
    # ==========================================
    
    role = Column(Enum(Role), default=Role.USER, nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Relasi
    uploaded_docs = relationship("Document", back_populates="uploader")
    favorites = relationship("Favorite", back_populates="user")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # 1. Metadata Utama (Sesuai UI Form Upload)
    title = Column(String, index=True, nullable=False)
    author = Column(String, index=True, nullable=False)
    author_degree = Column(String, nullable=False)
    publication_date = Column(Date, nullable=False)
    year = Column(Integer, index=True, nullable=False)
    language = Column(String, nullable=False)
    
    # 2. Kategori & Klasifikasi
    category = Column(Enum(DocumentCategory), nullable=False)
    prodi = Column(Enum(StudyProgram), index=True, nullable=False)
    tags = Column(ARRAY(String), default=[])
    
    # 3. Visibilitas & Status (Dropdown Form Upload)
    document_status = Column(Enum(VisibilityStatus), default=VisibilityStatus.PUBLIK, nullable=False)
    file_status = Column(Enum(VisibilityStatus), default=VisibilityStatus.PUBLIK, nullable=False)
    
    # 4. Konten Teks
    summary = Column(Text, nullable=True)   # Diisi otomatis oleh IndoT5 atau manual
    highlight = Column(Text, nullable=True) # Poin penting dokumen
    
    # 5. File Storage (Referensi ke MinIO)
    filename = Column(String, unique=True, index=True, nullable=False)
    file_url = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False) # Disimpan dalam bytes
    
    # 6. Tracking AI Pipeline & Watermark
    ai_status = Column(Enum(AIProcessingStatus), default=AIProcessingStatus.PENDING, nullable=False)
    is_watermarked = Column(Boolean, nullable=True) # Hasil deteksi CNN Blue Watermark
    watermark_confidence = Column(String, nullable=True) # Skor probabilitas dari CNN
    
    # 7. Statistik (Untuk Dashboard & Card)
    views_count = Column(Integer, default=0)
    downloads_count = Column(Integer, default=0)

    # 8. Sitasi Generator
    citation_mla = Column(Text, nullable=True)
    citation_apa = Column(Text, nullable=True)
    citation_ieee = Column(Text, nullable=True)
    citation_harvard = Column(Text, nullable=True)

    # 9. Audit & Relasi
    uploaded_by = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    uploader = relationship("User", back_populates="uploaded_docs")
    favorited_by = relationship("Favorite", back_populates="document", cascade="all, delete-orphan")


class Favorite(Base):
    __tablename__ = "favorites"
    
    # Mencegah 1 user memfavoritkan dokumen yang sama berkali-kali
    __table_args__ = (UniqueConstraint('user_id', 'document_id', name='_user_document_uc'),)

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relasi
    user = relationship("User", back_populates="favorites")
    document = relationship("Document", back_populates="favorited_by")
    
# 2. Tambahkan Model ini di bagian paling bawah file models.py
class Announcement(Base):
    __tablename__ = "announcements"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    type = Column(Enum(AnnouncementType), default=AnnouncementType.INFO, nullable=False)
    is_active = Column(Boolean, default=True) # Admin bisa menyembunyikan tanpa menghapus
    
    created_by = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    author = relationship("User")


class Guideline(Base):
    __tablename__ = "guidelines"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    step_number = Column(Integer, nullable=False, unique=True) # Urutan nomor langkah (1, 2, 3...)
    description = Column(String, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())