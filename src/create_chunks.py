from pathlib import Path
import json
import re
from tqdm import tqdm

IN_FILE = Path("data/processed/documents.jsonl")
OUT_FILE = Path("data/processed/chunks.jsonl")

def clean_for_chunk(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    text = clean_for_chunk(text)
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    total = 0
    with open(IN_FILE, "r", encoding="utf-8") as fin, open(OUT_FILE, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Chunk oluşturuluyor"):
            doc = json.loads(line)
            chunks = chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                    "doc_id": doc["doc_id"],
                    "source_type": doc["source_type"],
                    "period": doc["period"],
                    "legislative_year": doc["legislative_year"],
                    "session_no": doc["session_no"],
                    "filename": doc["filename"],
                    "chunk_index": i,
                    "text": chunk,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    print(f"Toplam chunk: {total}")
    print(f"Kaydedildi: {OUT_FILE}")

if __name__ == "__main__":
    main()
