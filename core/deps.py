# core/deps.py
from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.security import SECRET_KEY, ALGORITHM
from core.config import Config
from core.database import AsyncSessionLocal
from core.models import User

# Skema token: "Bearer eyJhbGci..."
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# 1. Fungsi untuk mendapatkan Database Session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency untuk inject session database ke endpoint."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# 2. Update fungsi validasi User (Sekarang butuh parameter db)
async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
):
    """
    Fungsi ini akan dipanggil di setiap endpoint yang butuh login.
    Tugasnya: Validasi Token & Ambil Data User dari DB via SQLAlchemy.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kredensial tidak valid",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode Token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Cek User di Database dengan format query SQLAlchemy 2.0
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    
    if user is None:
        raise credentials_exception
        
    if not user.active:
        raise HTTPException(status_code=400, detail="User tidak aktif")

    return user