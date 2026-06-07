import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

# Memaksa Python untuk membaca file .env di root folder
load_dotenv()

# Ambil DATABASE_URL murni hanya dari .env
DATABASE_URL = os.getenv("DATABASE_URL")

# Keamanan: Hentikan program jika DATABASE_URL tidak ditemukan
if not DATABASE_URL:
    raise ValueError("🚨 DATABASE_URL tidak ditemukan! Pastikan file .env sudah dikonfigurasi dengan benar.")

# Membuat Async Engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Membuat Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

Base = declarative_base()