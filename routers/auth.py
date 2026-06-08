# routers/auth.py
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel, EmailStr

# Import SQLAlchemy Database, Model, dan Security
from core.deps import get_db
from core.models import User, Role, StudyProgram
from core.security import get_password_hash, verify_password, create_access_token

# Import Fungsi Email yang baru dibuat
from core.email_utils import send_verification_email, send_reset_password_email

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ==========================================
# SCHEMAS (Validasi Input & Swagger UI)
# ==========================================
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    nim: str                   # Tambahan NIM
    prodi: StudyProgram        # Tambahan Prodi menggunakan Enum

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_name: str
    role: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

# ==========================================
# ENDPOINTS
# ==========================================


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Mendaftarkan user baru dan mengirim email verifikasi"""
    
    # --- 1. GATEKEEPER: Validasi Email Akademik PNJ ---
    # Memastikan email wajib diakhiri dengan ".pnj.ac.id"
    if not data.email.endswith(".pnj.ac.id"):
        raise HTTPException(
            status_code=400,
            detail="Registrasi ditolak. Anda wajib menggunakan email akademik resmi PNJ (berakhiran .pnj.ac.id)"
        )

    # 2. Cek apakah email sudah ada
    result = await db.execute(select(User).where(User.email == data.email))
    user_exist = result.scalars().first()
    
    if user_exist:
        raise HTTPException(
            status_code=400,
            detail="Email sudah terdaftar. Silakan gunakan email lain."
        )

    # Cek juga apakah NIM sudah terdaftar
    result_nim = await db.execute(select(User).where(User.nim == data.nim))
    nim_exist = result_nim.scalars().first()
    if nim_exist:
        raise HTTPException(
            status_code=400,
            detail="NIM sudah terdaftar di sistem."
        )

    # 3. Hash Password & Generate Token Verifikasi
    hashed_pwd = get_password_hash(data.password)
    verification_token = secrets.token_urlsafe(32)

    # 4. Simpan ke Database
    new_user = User(
        email=data.email,
        password_hash=hashed_pwd,
        full_name=data.full_name,
        nim=data.nim,            # Masukkan NIM dari payload
        prodi=data.prodi,        # Masukkan Prodi dari payload
        role=Role.USER, 
        active=True,
        is_verified=False, # User belum bisa login sebelum verifikasi email
        verification_token=verification_token
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # 5. Kirim email verifikasi di background
    await send_verification_email(data.email, verification_token)

    return {"message": "Registrasi berhasil. Silakan cek email Anda untuk verifikasi."}


# @router.post("/register", status_code=status.HTTP_201_CREATED)
# async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
#     """Mendaftarkan user baru dan mengirim email verifikasi"""
    
#     # 1. Cek apakah email sudah ada
#     result = await db.execute(select(User).where(User.email == data.email))
#     user_exist = result.scalars().first()
    
#     if user_exist:
#         raise HTTPException(
#             status_code=400,
#             detail="Email sudah terdaftar. Silakan gunakan email lain."
#         )

#     # Cek juga apakah NIM sudah terdaftar
#     result_nim = await db.execute(select(User).where(User.nim == data.nim))
#     nim_exist = result_nim.scalars().first()
#     if nim_exist:
#         raise HTTPException(
#             status_code=400,
#             detail="NIM sudah terdaftar di sistem."
#         )

#     # 2. Hash Password & Generate Token Verifikasi
#     hashed_pwd = get_password_hash(data.password)
#     verification_token = secrets.token_urlsafe(32)

#     # 3. Simpan ke Database
#     new_user = User(
#         email=data.email,
#         password_hash=hashed_pwd,
#         full_name=data.full_name,
#         nim=data.nim,            # Masukkan NIM dari payload
#         prodi=data.prodi,        # Masukkan Prodi dari payload
#         role=Role.USER, 
#         active=True,
#         is_verified=False, # User belum bisa login sebelum verifikasi email
#         verification_token=verification_token
#     )
    
#     db.add(new_user)
#     await db.commit()
#     await db.refresh(new_user)

#     # 4. Kirim email verifikasi di background
#     await send_verification_email(data.email, verification_token)

#     return {"message": "Registrasi berhasil. Silakan cek email Anda untuk verifikasi."}


@router.get("/verify")
async def verify_email(token: str, db: AsyncSession = Depends(get_db)):
    print("\n=== MULAI DEBUGGING VERIFIKASI ===")
    print(f"1. Token dari Postman : '{token}'") 
    
    # Intip semua data user yang ada di database menurut FastAPI
    result_all = await db.execute(select(User))
    users_in_db = result_all.scalars().all()
    print(f"2. Total user di DB   : {len(users_in_db)} orang")
    
    for u in users_in_db:
        print(f"   -> Cek User: {u.full_name} | Token di DB: '{u.verification_token}'")
        if str(u.verification_token) == str(token):
            print("      [!] KETEMU! Token cocok secara teks di Python!")

    # Logika eksekusi query asli
    result = await db.execute(select(User).where(User.verification_token == token))
    user = result.scalars().first()
    
    print(f"3. Hasil Query Filter : {user}")
    print("=== SELESAI DEBUGGING ===\n")
    
    if not user:
        raise HTTPException(status_code=400, detail="Token verifikasi tidak valid atau sudah digunakan.")
        
    user.is_verified = True
    user.verification_token = None
    await db.commit()
    
    return {"message": "Email berhasil diverifikasi! Anda sekarang dapat login."}

@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Login user dan kembalikan JWT Token"""
    
    # 1. Cari User by Email
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=400, detail="Email atau password salah")

    # 2. Cek Password
    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Email atau password salah")

    # 3. Cek apakah akun aktif dan email terverifikasi
    if not user.active:
        raise HTTPException(status_code=403, detail="Akun Anda dinonaktifkan.")
        
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Silakan verifikasi email Anda terlebih dahulu.")

    # 4. Buat Token
    access_token = create_access_token(data={"sub": user.email, "role": user.role.value})

    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user_name": user.full_name,
        "role": user.role.value
    }


@router.post("/forgot-password")
async def forgot_password(data: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Generate token reset password dan kirim link via email"""
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalars().first()
    
    if user:
        # Generate token unik dan set waktu kadaluarsa 15 menit
        reset_token = secrets.token_urlsafe(32)
        user.reset_password_token = reset_token
        user.reset_password_expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
        await db.commit()
        
        # Tembak via SMTP
        await send_reset_password_email(data.email, reset_token)

    return {"message": "Jika email terdaftar di sistem kami, link reset password telah dikirim."}


@router.post("/reset-password")
async def reset_password(data: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Ubah password menggunakan token dari email"""
    result = await db.execute(select(User).where(User.reset_password_token == data.token))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=400, detail="Token reset password tidak valid.")
        
    # Validasi batas waktu (15 Menit) menggunakan zona UTC yang aman
    now_utc = datetime.now(timezone.utc)
    expires_at = user.reset_password_expires_at
    
    # Memastikan data waktu dari database kompatibel untuk dibandingkan
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
        
    if now_utc > expires_at:
        raise HTTPException(status_code=400, detail="Token reset password sudah kedaluwarsa (lewat 15 menit).")
        
    # Ubah ke password baru
    user.password_hash = get_password_hash(data.new_password)
    
    # Hancurkan token agar tidak bisa disalahgunakan lagi
    user.reset_password_token = None
    user.reset_password_expires_at = None
    await db.commit()
    
    return {"message": "Password berhasil diubah. Silakan login menggunakan password baru."}