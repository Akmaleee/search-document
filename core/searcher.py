from elasticsearch import Elasticsearch
import re
from typing import List, Dict, Any
from core.config import Config

es = Elasticsearch(Config.ELASTICSEARCH_URL)

def clean_text(text: str) -> str:
    """Hapus karakter newline dari teks highlight."""
    # Ganti newline dan spasi ganda jadi satu spasi
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def keyword_search(query: str, size: int = 5) -> List[Dict[str, Any]]:
    """Cari berdasarkan kata kunci di Elasticsearch, highlight bersih dan rapi."""
    result = es.search(
        index="partnership",
        query={"match": {"content": query}},
        highlight={"fields": {"content": {}}},
        size=size
    )

    hits = []
    for h in result["hits"]["hits"]:
        highlight_list = h.get("highlight", {}).get("content", [])
        if highlight_list:
            # Bersihkan setiap kalimat hasil highlight
            highlights = [{"sentence": clean_text(text)} for text in highlight_list]
        else:
            highlights = [{"sentence": clean_text(h["_source"]["content"][:200])}]

        hits.append({
            "id": h["_id"],
            "score": h["_score"],
            "path": h["_source"].get("path"),
            "highlight": highlights
        })

    return hits
