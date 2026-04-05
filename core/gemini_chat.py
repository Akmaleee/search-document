# core/gemini_chat.py
from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv
from helpers.prompt import METADATA_EXTRACTION_PROMPT  # Pastikan ini di-import

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_answer(prompt: str, webSearch: bool = False):
    """
    Generate jawaban berbasis konteks menggunakan model Gemini.
    Jika webSearch=True, maka Google Search akan digunakan sebagai sumber tambahan.
    """

    # Jika webSearch True, tambahkan grounding tool Google Search
    if webSearch:
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
    else:
        # Tanpa Google Search
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

    return response.text

def extract_metadata_with_gemini(text_content: str):
    """
    Mengirim teks dokumen ke Gemini untuk dianalisis metadatanya
    (Tahun, Tags, Summary, Citations) sesuai format JSON.
    """
    try:
        # Potong teks jika terlalu panjang (hemat token & agar fokus ke header/intro dokumen)
        # Ambil 15.000 karakter pertama yang biasanya memuat judul, abstrak, dan intro
        truncated_text = text_content[:15000]

        full_prompt = f"{METADATA_EXTRACTION_PROMPT}\n\n=== DOKUMEN ===\n{truncated_text}"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json" # Memaksa output JSON agar lebih stabil
            )
        )

        # Bersihkan response (hapus backticks ```json ... ``` jika masih ada)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        
        # Parsing string JSON ke Dictionary Python
        metadata = json.loads(clean_json)
        
        return metadata

    except Exception as e:
        print(f"❌ Gagal ekstraksi metadata AI: {e}")
        # Return fallback jika gagal agar sistem tidak crash
        return {
            "year": None,
            "tags": ["Uncategorized"],
            "summary": "Gagal membuat ringkasan otomatis.",
            "citations": {}
        }

# from core.config import Config
# from google import genai
# from google.genai import types
# import os
# from dotenv import load_dotenv
# load_dotenv()

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# def generate_answer(prompt: str, webSearch: bool = False):
#     """
#     Generate jawaban berbasis konteks menggunakan model Gemini.
#     Jika webSearch=True, maka Google Search akan digunakan sebagai sumber tambahan.
#     """

#     # Jika webSearch True, tambahkan grounding tool Google Search
#     if webSearch:
#         grounding_tool = types.Tool(
#             google_search=types.GoogleSearch()
#         )

#         config = types.GenerateContentConfig(
#             tools=[grounding_tool]
#         )

#         response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=prompt,
#             config=config,
#         )
#     else:
#         # Tanpa Google Search
#         response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=prompt,
#         )

#     return response.text
