from core.config import Config
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
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
