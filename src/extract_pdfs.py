from pathlib import Path
import json
import re
import fitz
from tqdm import tqdm

RAW_DIR = Path("data/raw/tutanaklar")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "documents.jsonl"

def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pdf(pdf_path: Path):
    pdf = fitz.open(str(pdf_path))
    pages = []
    for page_no, page in enumerate(pdf, start=1):
        text = clean_text(page.get_text("text"))
        if text:
            pages.append({"page": page_no, "text": text})
    return pages

def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"), key=lambda p: int(p.stem))
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for pdf_path in tqdm(pdf_files, desc="PDF metni çıkarılıyor"):
            session_no = int(pdf_path.stem)
            pages = extract_pdf(pdf_path)
            full_text = "\n\n".join(p["text"] for p in pages)
            record = {
                "doc_id": f"tbmm_28_3_{session_no}",
                "source_type": "genel_kurul_tutanagi",
                "period": 28,
                "legislative_year": 3,
                "session_no": session_no,
                "filename": pdf_path.name,
                "page_count": len(pages),
                "text": full_text,
                "pages": pages,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Kaydedildi: {OUT_FILE}")

if __name__ == "__main__":
    main()
