import asyncio
from prisma import Prisma
from passlib.context import CryptContext

# Setup hashing password
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def main():
    db = Prisma()
    await db.connect()

    print("🌱 Seeding database...")

    # Data Admin Default
    email = "admin@gmail.co.id"
    password = "admin123" 
    hashed_password = pwd_context.hash(password)

    # Cek apakah user sudah ada
    existing_user = await db.user.find_unique(where={"email": email})

    if not existing_user:
        user = await db.user.create(
            data={
                "email": email,
                "password_hash": hashed_password,
                "full_name": "Super Admin",
                "role": "ADMIN"
            }
        )
        print(f"✅ User Admin created: {user.email} | Password: {password}")
    else:
        print(f"⚠️ User {email} already exists.")

    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())