# core/deps.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from prisma import Prisma
from core.security import SECRET_KEY, ALGORITHM
from core.config import Config

# Skema token: "Bearer eyJhbGci..."
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

db = Prisma()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Fungsi ini akan dipanggil di setiap endpoint yang butuh login.
    Tugasnya: Validasi Token & Ambil Data User dari DB.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kredensial tidak valid",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # 1. Decode Token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # 2. Cek User di Database
    if not db.is_connected():
        await db.connect()
        
    user = await db.user.find_unique(where={"email": email})
    
    if user is None:
        raise credentials_exception
        
    if not user.active:
        raise HTTPException(status_code=400, detail="User tidak aktif")

    return user