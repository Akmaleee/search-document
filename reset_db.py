import asyncio
from core.database import engine, Base
# Import semua model agar SQLAlchemy mengenalinya sebelum didrop/dicreate
from core.models import User, Document, Favorite, Role, DocumentCategory, StudyProgram, VisibilityStatus, AIProcessingStatus

async def reset_database():
    print("⚠️ Memulai proses reset database...")
    async with engine.begin() as conn:
        print("🗑️ Menghapus tabel lama...")
        # Drop all akan menghapus semua tabel yang terkait dengan Base
        await conn.run_sync(Base.metadata.drop_all)
        
        print("✨ Membuat tabel baru dengan skema terbaru...")
        await conn.run_sync(Base.metadata.create_all)
        
    print("✅ Database berhasil di-reset ke skema terbaru!")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(reset_database())