import cv2
import numpy as np
import fitz  # PyMuPDF

def detect_pnj_watermark_in_image(img_bgr: np.ndarray) -> bool:
    """
    Fungsi inti untuk mendeteksi warna Cyan/Biru PNJ menggunakan HSV.
    """
    # 1. Konversi BGR (format default OpenCV) ke HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. Definisikan Range Warna Watermark PNJ (Cyan / Light Blue)
    # Di OpenCV: Hue range 0-179. Cyan sekitar 90.
    # Karena watermarknya transparan (pudar), Saturation tidak terlalu tinggi, dan Value cukup terang (latar putih).
    lower_blue = np.array([80, 15, 150])   # Batas bawah (Biru Kehijauan pudar)
    upper_blue = np.array([110, 150, 255]) # Batas atas (Biru terang)
    
    # 3. Buat Masking (Saring hanya pixel yang masuk range di atas)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 4. Hitung persentase pixel watermark dibanding total pixel gambar
    watermark_pixels = cv2.countNonZero(mask)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    
    ratio = (watermark_pixels / total_pixels) * 100
    
    # 5. Threshold logika: Jika lebih dari 0.5% area halaman adalah warna biru PNJ, 
    # maka kita asumsikan halaman tersebut memiliki watermark.
    # (Nilai 0.5 ini bisa kamu naik-turunkan saat testing nanti)
    if ratio > 0.5:
        return True
    return False

def check_document_watermark(file_bytes: bytes, filename: str) -> bool:
    """
    Menerima file (PDF atau Gambar), memprosesnya, dan mengecek watermark.
    """
    is_watermarked = False
    
    # --- SKENARIO 1: JIKA FILE ADALAH PDF ---
    if filename.lower().endswith('.pdf'):
        # Buka PDF langsung dari memory (tanpa save ke disk dulu)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Cek maksimal 3 halaman pertama saja agar server tidak berat
        pages_to_check = min(3, len(doc))
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            # Render halaman ke matrix pixel
            pix = page.get_pixmap(dpi=150) 
            
            # Konversi pixel dari PyMuPDF ke format Numpy Array milik OpenCV
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Konversi format warna (biasanya PyMuPDF outputnya RGB atau RGBA)
            if pix.n == 4: # Punya channel Alpha (Transparan)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else: # RGB standar
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
            # Lakukan deteksi
            if detect_pnj_watermark_in_image(img_bgr):
                is_watermarked = True
                break # Langsung berhenti kalau ketemu di salah satu halaman
        
        doc.close()
        
    # --- SKENARIO 2: JIKA FILE ADALAH GAMBAR (JPG/PNG) ---
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            is_watermarked = detect_pnj_watermark_in_image(img_bgr)
            
    return is_watermarked