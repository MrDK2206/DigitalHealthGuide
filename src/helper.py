import os
from typing import List, Dict
from dotenv import load_dotenv
from pypdf import PdfReader
import hashlib
import math
import re


load_dotenv()


def load_pdf_file(data_dir: str) -> List[Dict]:
    """Load PDFs from a directory and return list of docs with text and metadata."""
    docs = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fname)
            try:
                reader = PdfReader(path)
                text = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text.append(page_text)
                full_text = "\n".join(text)
                docs.append({"text": full_text, "metadata": {"source": fname, "path": path}})
            except Exception:
                # skip unreadable PDFs
                continue
    return docs


def text_split(docs: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Split documents into chunks with metadata preserved."""
    chunks = []
    for doc in docs:
        text = doc.get("text", "")
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "metadata": doc.get("metadata", {})})
            start = end - overlap
            if start < 0:
                start = 0
    return chunks


def _local_embedding(text: str, dimensions: int = 384) -> List[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    for token in tokens:
        index = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % dimensions
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]

    return vector


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Return deterministic local embeddings for a list of texts."""
    return [_local_embedding(text) for text in texts]


def build_id(text: str, metadata: Dict) -> str:
    content = text + repr(sorted(metadata.items()))
    return hashlib.sha1(content.encode("utf-8")).hexdigest()