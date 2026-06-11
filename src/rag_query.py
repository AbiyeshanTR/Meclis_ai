from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer

VECTOR_DIR = Path("data/vector_db")
MODEL_NAME = "intfloat/multilingual-e5-base"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(str(VECTOR_DIR / "faiss.index"))

with open(VECTOR_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def search(query: str, top_k: int = 5):
    q_emb = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        item = chunks[int(idx)]
        results.append({
            "score": float(score),
            "session_no": item["session_no"],
            "filename": item["filename"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
        })
    return results

if __name__ == "__main__":
    query = input("Soru: ")
    for i, r in enumerate(search(query, top_k=5), start=1):
        print("\n" + "=" * 80)
        print(f"[{i}] Skor: {r['score']:.3f} | Birleşim: {r['session_no']} | Dosya: {r['filename']}")
        print(r["text"][:1000])
