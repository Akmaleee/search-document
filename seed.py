import asyncio
from passlib.context import CryptContext
from sqlalchemy.future import select
from core.database import AsyncSessionLocal
from core.models import User, Role, StudyProgram

# Setup hashing password
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def main():
    print("🌱 Seeding database...")

    # Data Admin Default
    email = "pointblankmalware@gmail.com"
    password = "admin123"
    hashed_password = pwd_context.hash(password)

    # Gunakan session secara aman lewat AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        # Cek apakah user sudah ada
        result = await db.execute(select(User).where(User.email == email))
        existing_user = result.scalars().first()

        if not existing_user:
            new_admin = User(
                email=email,
                password_hash=hashed_password,
                full_name="Akmal Nur Wahid", # Nama disesuaikan
                
                # Profil Akademik
                nim="2207411047", # NIM disesuaikan
                prodi=StudyProgram.TI, 
                
                # Status keamanan & akun
                is_verified=True, 
                role=Role.ADMIN,
                active=True
            )
            db.add(new_admin)
            await db.commit()
            print(f"✅ User Admin created: {new_admin.email} | Name: {new_admin.full_name} | NIM: {new_admin.nim}")
        else:
            print(f"⚠️ User {email} already exists.")

if __name__ == "__main__":
    asyncio.run(main())