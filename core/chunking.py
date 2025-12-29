from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.config import Config

def chunk_text(text: str, chunk_size: int = None, overlap: int = None):
    """
    Potong teks menjadi potongan kecil (chunks) menggunakan LangChain RecursiveCharacterTextSplitter.
    Ini menjaga pemotongan tetap alami — tidak memotong di tengah kalimat/paragraf.
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # prioritas pemisahan
    )

    chunks = splitter.split_text(text)
    return chunks
