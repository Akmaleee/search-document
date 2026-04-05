# routers/auth.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from prisma import Prisma
from core.security import get_password_hash, verify_password, create_access_token
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["Authentication"])
db = Prisma() # Instance DB sementara (akan connect via main.py)

# --- Schemas (Validasi Input) ---
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_name: str
    role: str

# --- Endpoints ---

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(data: RegisterRequest):
    """Mendaftarkan user baru"""
    if not db.is_connected():
        await db.connect()

    # 1. Cek apakah email sudah ada
    user_exist = await db.user.find_unique(where={"email": data.email})
    if user_exist:
        raise HTTPException(
            status_code=400,
            detail="Email sudah terdaftar. Silakan gunakan email lain."
        )

    # 2. Hash Password
    hashed_pwd = get_password_hash(data.password)

    # 3. Simpan ke Database
    new_user = await db.user.create(
        data={
            "email": data.email,
            "password_hash": hashed_pwd,
            "full_name": data.full_name,
            "role": "USER",      # Default role
            "active": True
        }
    )

    return {"message": "User berhasil didaftarkan", "email": new_user.email}

@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest):
    """Login user dan kembalikan JWT Token"""
    if not db.is_connected():
        await db.connect()

    # 1. Cari User by Email
    user = await db.user.find_unique(where={"email": data.email})
    if not user:
        raise HTTPException(status_code=400, detail="Email atau password salah")

    # 2. Cek Password
    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Email atau password salah")

    # 3. Cek apakah aktif
    if not user.active:
        raise HTTPException(status_code=400, detail="Akun Anda dinonaktifkan.")

    # 4. Buat Token
    access_token = create_access_token(data={"sub": user.email, "role": str(user.role)})

    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user_name": user.full_name,
        "role": user.role
    }