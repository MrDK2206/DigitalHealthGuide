import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from src.helper import load_pdf_file, text_split, embed_texts, build_id


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"


def get_env_value(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value

    joined_names = ", ".join(names)
    raise RuntimeError(f"Missing required environment variable. Set one of: {joined_names}")


load_dotenv()

PINECONE_API_KEY = get_env_value("PINECONE_API_KEY", "PINECONE_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

data_directory = DATA_DIR
if not data_directory.exists():
    raise FileNotFoundError("The Data/ directory is missing. Add the source PDFs before building the index.")

docs = load_pdf_file(data_dir=str(data_directory))
if not docs:
    raise RuntimeError("No PDF documents were found in Data/.")

chunks = text_split(docs)
texts = [c["text"] for c in chunks]
metadatas = [c.get("metadata", {}) for c in chunks]

embeddings = embed_texts(texts)

index_name = os.getenv("PINECONE_INDEX_NAME", "medicalbot")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing = pc.list_indexes().names()

if index_name not in existing:
    dim = len(embeddings[0]) if embeddings else 1536
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    if pinecone_env:
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=pinecone_env),
        )
    else:
        pc.create_index(name=index_name, dimension=dim, metric="cosine")


def wait_for_index_ready(name: str, poll_interval: float = 2.0, timeout_seconds: int = 300):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        info = pc.describe_index(name=name)
        status = info.get("status", {}) if isinstance(info, dict) else getattr(info, "status", {})
        ready = status.get("ready", False) if isinstance(status, dict) else getattr(status, "ready", False)
        state = status.get("state", "Unknown") if isinstance(status, dict) else getattr(status, "state", "Unknown")
        if ready:
            host = info.get("host") if isinstance(info, dict) else getattr(info, "host", None)
            if host:
                return host
            break

        print(f"Waiting for Pinecone index '{name}' to be ready... current state: {state}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Pinecone index '{name}' was not ready after {timeout_seconds} seconds.")

index_host = wait_for_index_ready(index_name)

# get index and upsert
index = pc.Index(host=index_host)

# prepare vectors and upsert in batches
batch_size = 100
vectors = []
for idx, (emb, meta, txt) in enumerate(zip(embeddings, metadatas, texts)):
    vid = build_id(txt, meta)
    # store a short preview in metadata to keep result compact
    meta_copy = dict(meta)
    meta_copy["text_preview"] = txt[:400]
    vectors.append((vid, emb, meta_copy))

for i in range(0, len(vectors), batch_size):
    batch = vectors[i : i + batch_size]
    print(f"Upserting vectors {i + 1}-{i + len(batch)} of {len(vectors)}...")
    index.upsert(vectors=batch)

print(f"Finished indexing {len(vectors)} chunks into Pinecone index '{index_name}'.")
