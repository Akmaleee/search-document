import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from core.config import Config

def send_email_sync(to_email: str, subject: str, html_body: str):
    """Fungsi dasar pengirim email via SMTP (Synchronous)"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = Config.SMTP_USERNAME
    msg["To"] = to_email

    part = MIMEText(html_body, "html")
    msg.attach(part)

    # Koneksi ke SMTP Gmail
    with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
        server.starttls() # Mengamankan koneksi
        server.login(Config.SMTP_USERNAME, Config.SMTP_CODE)
        server.sendmail(Config.SMTP_USERNAME, to_email, msg.as_string())

async def send_email_async(to_email: str, subject: str, html_body: str):
    """Membungkus fungsi sync menjadi async background task"""
    await asyncio.to_thread(send_email_sync, to_email, subject, html_body)

async def send_verification_email(to_email: str, token: str):
    verify_url = f"{Config.FRONTEND_URL}/verify?token={token}"
    subject = "Verifikasi Akun Smart Repository"
    html = f"""
    <h3>Selamat Datang di Smart Repository!</h3>
    <p>Terima kasih telah mendaftar. Silakan klik tombol di bawah ini untuk memverifikasi email Anda:</p>
    <a href="{verify_url}" style="padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px;">Verifikasi Akun</a>
    <p>Jika tombol tidak berfungsi, salin dan tempel link berikut ke browser Anda: <br> {verify_url}</p>
    """
    await send_email_async(to_email, subject, html)

async def send_reset_password_email(to_email: str, token: str):
    reset_url = f"{Config.FRONTEND_URL}/reset-password?token={token}"
    subject = "Permintaan Reset Password Smart Repository"
    html = f"""
    <h3>Reset Password Anda</h3>
    <p>Kami menerima permintaan untuk mereset password akun Anda. Klik tombol di bawah ini untuk melanjutkan:</p>
    <a href="{reset_url}" style="padding: 10px 20px; background-color: #dc3545; color: white; text-decoration: none; border-radius: 5px;">Reset Password</a>
    <p><b>Penting:</b> Link ini hanya berlaku selama 15 menit. Jika Anda tidak merasa meminta reset password, abaikan email ini.</p>
    """
    await send_email_async(to_email, subject, html)