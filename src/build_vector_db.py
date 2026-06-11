from pathlib import Path
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
VECTOR_DIR = Path("data/vector_db")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-base"

def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    chunks = load_chunks()
    texts = ["passage: " + c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(VECTOR_DIR / "faiss.index"))

    with open(VECTOR_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index kaydedildi: {VECTOR_DIR / 'faiss.index'}")
    print(f"Chunk metadata kaydedildi: {VECTOR_DIR / 'chunks.pkl'}")

if __name__ == "__main__":
    main()
