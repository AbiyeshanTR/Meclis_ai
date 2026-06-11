import sys
import json
import subprocess
from pathlib import Path


REQUIRED_PACKAGES = [
    "numpy",
    "requests",
    "sentence-transformers",
    "faiss-cpu"
]


def install_packages():
    print("\n[1] Gerekli Python paketleri kontrol ediliyor...")

    for package in REQUIRED_PACKAGES:
        try:
            import_name = package.replace("-", "_")

            if package == "faiss-cpu":
                import_name = "faiss"
            elif package == "sentence-transformers":
                import_name = "sentence_transformers"

            __import__(import_name)
            print(f"Zaten kurulu: {package}")

        except ImportError:
            print(f"Kuruluyor: {package}")
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                package
            ])


install_packages()



import requests
import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"

VECTOR_DIR = BASE_DIR / "data" / "vector_db"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_FILE = VECTOR_DIR / "faiss.index"
METADATA_FILE = VECTOR_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Ollama modeli
OLLAMA_MODEL = "qwen2.5:7b"

# Arama ayarları
REWRITE_QUERY_COUNT = 5
RAW_TOP_K = 12
FINAL_SOURCE_COUNT = 8



# 1. Ollama yardımcı fonksiyonu


def call_ollama(prompt, temperature=0.0, num_ctx=8192, timeout=240):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "top_p": 0.8,
                "repeat_penalty": 1.15
            }
        },
        timeout=timeout
    )

    response.raise_for_status()
    return response.json()["response"]



# 2. Chunk verisini oku


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"chunks.jsonl bulunamadı: {CHUNKS_FILE}")

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"\n[2] Toplam chunk sayısı: {len(chunks)}")
    return chunks



# 3. FAISS vector database oluştur


def build_vector_db(chunks):
    print("\n[3] Embedding modeli yükleniyor...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("\n[4] Chunk embedding'leri oluşturuluyor...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_FILE))

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n[5] Vector database oluşturuldu.")
    print(f"FAISS index: {FAISS_INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")



# 4. Var olan vector database'i yükle


def load_vector_db():
    index = faiss.read_index(str(FAISS_INDEX_FILE))

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata



# 5. LLM ile sorgu yeniden yazma


def rewrite_question_with_ollama(question):
    prompt = f"""
SADECE TÜRKÇE YAZ.
İngilizce, Çince veya başka bir dil kullanma.

Sen TBMM Genel Kurul tutanakları üzerinde çalışan bir arama sorgusu üreticisisin.

Kullanıcı sorusu:
{question}

Görevin:
Bu soruyu TBMM tutanaklarında semantik arama yapmak için {REWRITE_QUERY_COUNT} farklı Türkçe arama sorgusuna dönüştür.

Kurallar:
- Sadece arama sorgularını yaz.
- Açıklama yapma.
- Numara koyma.
- Her satıra bir sorgu yaz.
- Yazım hatalarını düzelt.
- Türkçe karakterleri doğru kullan.
- Parti, kurum, kişi ve konu adlarını doğru biçimde yaz.
- Sorunun anlamını değiştirme.
- Eğer soru bir partiye, kişiye veya kuruma yöneltilen eleştirileri soruyorsa, yönü koru.
- Eğer soru "X hakkında gündem nedir" diyorsa, X ile ilgili gündem başlıklarını bulduracak sorgular üret.
- Eğer soru "X hakkında bahsedilen konular" diyorsa, X ile ilgili konuşulan konu başlıklarını bulduracak sorgular üret.

Örnek:
Sağlık alanında gündem nedir
Sağlık sistemi hakkında konuşulan başlıklar
Sağlık hizmetleriyle ilgili eleştiriler
Sağlık kanun teklifi hakkında görüşmeler
Hastane, randevu ve sağlık çalışanları hakkında gündem

Şimdi sadece sorguları üret:
"""

    try:
        text = call_ollama(
            prompt=prompt,
            temperature=0.0,
            num_ctx=2048,
            timeout=120
        )

        queries = []

        for line in text.splitlines():
            line = line.strip()
            line = line.lstrip("-").strip()
            line = line.lstrip("•").strip()

            if len(line) > 2 and line[0].isdigit():
                line = line.lstrip("0123456789").lstrip(".").lstrip(")").strip()

            if line and len(line) > 3:
                queries.append(line)

        queries.insert(0, question)

        unique_queries = []
        for q in queries:
            if q not in unique_queries:
                unique_queries.append(q)

        return unique_queries[:REWRITE_QUERY_COUNT + 1]

    except Exception as e:
        print(f"\nSorgu genişletme başarısız oldu. Orijinal soru ile devam ediliyor: {e}")
        return [question]


# 6. Çoklu sorgu ile FAISS arama


def search_chunks(question, index, metadata, model, top_k=RAW_TOP_K):
    search_queries = rewrite_question_with_ollama(question)

    print("\nÜretilen arama sorguları:")
    for q in search_queries:
        print(f"- {q}")

    all_results = {}

    for search_query in search_queries:
        query_embedding = model.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = index.search(query_embedding, top_k)

        for score, idx in zip(scores[0], indices[0]):
            item = dict(metadata[idx])
            item["score"] = float(score)

            chunk_id = item.get("chunk_id", str(idx))

            if chunk_id not in all_results:
                all_results[chunk_id] = item
            else:
                if item["score"] > all_results[chunk_id]["score"]:
                    all_results[chunk_id] = item

    results = list(all_results.values())
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]



# 7. Kaynakları yazdır


def print_retrieved_sources(results, title="Bulunan kaynaklar"):
    print(f"\n{title}:")
    print("=" * 80)

    for i, r in enumerate(results, start=1):
        print(
            f"[{i}] Skor: {r['score']:.3f} | "
            f"Birleşim: {r.get('session_no')} | "
            f"Dosya: {r.get('filename')}"
        )

        preview = r["text"][:900].replace("\n", " ")
        print(preview)
        print("-" * 80)



# 8. Final cevap için context oluştur


def build_context(retrieved_chunks):
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source_header = (
            f"[{i}] Birleşim: {chunk.get('session_no')} | "
            f"Dosya: {chunk.get('filename')} | "
            f"Skor: {chunk.get('score'):.3f}"
        )

        text = chunk["text"]

        if len(text) > 1600:
            text = text[:1600]

        context_parts.append(source_header + "\n" + text)

    separator = "\n\n" + "=" * 80 + "\n\n"
    return separator.join(context_parts)


# ============================================================
# 12. Final cevap üret
# ============================================================

def ask_ollama_final_answer(question, retrieved_chunks):
    context = build_context(retrieved_chunks)

    prompt = f"""
SADECE TÜRKÇE CEVAP VER.
İngilizce, Çince veya başka bir dil kullanma.
Bozuk, anlamsız veya yarım kelime kullanma.

Sen TBMM Genel Kurul tutanakları üzerinde çalışan tarafsız Türkçe bir analiz asistanısın.

Aşağıdaki kaynak parçaları TBMM tutanaklarından alınmıştır.
Kullanıcının sorusuna SADECE bu kaynak parçalarına dayanarak cevap ver.

Kullanıcı sorusu:
{question}

Kurallar:
1. Kaynaklarda olmayan bilgi ekleme.
2. Kişi, parti veya kurum adı uydurma.
3. Cevabı tarafsız, açık ve rapor diliyle yaz.
4. Alıntı yapma; kaynaklardaki bilgileri özetle.
5. Her önemli maddenin sonunda kaynak numarası kullan: [1], [2] gibi.
6. Eğer soru "X hakkında gündem nedir?" ise X ile ilgili gündem başlıklarını listele.
7. Eğer soru "X hakkında bahsedilen konular nelerdir?" ise X hakkında geçen farklı konu başlıklarını listele.
8. Eğer soru "X'e yapılan eleştiriler nelerdir?" ise X'in yaptığı eleştirileri değil, X'e yöneltilen eleştirileri özetle.
9. Eğer soru çelişki, doğruluk, yalan, haklılık gibi mantıksal karşılaştırma gerektiriyorsa ve kaynaklar doğrudan yeterli değilse bunu açıkça belirt.
10. Kaynaklarda yeterli doğrudan bilgi yoksa "Kaynak parçalarında bu soruya yeterli doğrudan bilgi yoktur." de.
11. Kaynaklarda birden fazla farklı başlık varsa en az 5 madde yaz.
12. Sadece ilk 2-3 kaynağa bağlı kalma; mümkünse verilen tüm kaynaklardaki farklı başlıkları kapsa.
13. Aynı anlama gelen kaynakları birleştir, farklı gündemleri ayrı madde yap.
14. En sonda "Kaynaklar" başlığı altında kullandığın kaynakları listele.

Kaynak parçaları:
{context}

Cevap formatı kesinlikle şöyle olmalı:

Kaynaklara göre ...

1. ...
2. ...
3. ...

Kaynaklar:
[1] Birleşim ..., Dosya ...
[2] Birleşim ..., Dosya ...
"""

    try:
        return call_ollama(
            prompt=prompt,
            temperature=0.0,
            num_ctx=8192,
            timeout=240
        )

    except requests.exceptions.ConnectionError:
        return """
Ollama bağlantısı kurulamadı.

Kontrol et:
1. Ollama kurulu mu?
2. Ollama açık mı?
3. qwen2.5:7b modeli yüklü mü?

PowerShell'de test:
ollama run qwen2.5:7b
"""

    except requests.exceptions.HTTPError as e:
        return f"""
Ollama HTTP hatası verdi:

{e}

Kullanılan model:
{OLLAMA_MODEL}

Model yüklü değilse:
ollama pull qwen2.5:7b
"""

    except Exception as e:
        return f"Ollama cevap üretirken beklenmeyen hata verdi: {e}"


# ============================================================
# 13. Ana program
# ============================================================

def main():
    print("\nMeclisAI RAG sistemi başlatılıyor...")
    print(f"Kullanılan LLM modeli: {OLLAMA_MODEL}")

    chunks = load_chunks()

    if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("\nVector database bulunamadı. İlk kez oluşturulacak.")
        build_vector_db(chunks)
    else:
        print("\nVar olan vector database bulundu. Yeniden oluşturulmayacak.")

    print("\n[6] Vector database yükleniyor...")
    index, metadata = load_vector_db()

    print("\n[7] Embedding modeli yükleniyor...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\nSistem hazır.")
    print("Çıkmak için: q")
    print("Örnek sorular:")
    print("- Sağlık alanında gündem nedir?")
    print("- Enerji politikaları hakkında gündem var mı?")
    print("- Recep Tayyip Erdoğan hakkında bahsedilen konular nelerdir?")
    print("- Cumhuriyet Halk Partisine yapılan eleştiriler nelerdir?")
    print("- AK Parti'ye yöneltilen eleştiriler nelerdir?")
    print("- Tarım konusunda neler konuşulmuş?")
    print("- Zeytinlikler ve maden yasası hakkında ne söylenmiş?")
    print("-" * 80)

    while True:
        question = input("\nSorunu yaz: ").strip()

        if question.lower() in ["q", "quit", "exit", "çık", "cik"]:
            print("Çıkılıyor...")
            break

        if not question:
            continue

        print("\n[1] Soru LLM ile arama sorgularına dönüştürülüyor...")
        print("[2] İlgili kaynaklar FAISS ile aranıyor...")

        results = search_chunks(
            question=question,
            index=index,
            metadata=metadata,
            model=embedding_model,
            top_k=RAW_TOP_K
        )

        final_sources = results[:FINAL_SOURCE_COUNT]

        print_retrieved_sources(
            final_sources,
            title="Final cevapta kullanılacak kaynaklar"
        )

        print("\n[3] Final cevap üretiliyor...")

        answer = ask_ollama_final_answer(
            question=question,
            retrieved_chunks=final_sources
        )

        print("\nCEVAP:")
        print("=" * 80)
        print(answer)
        print("=" * 80)


if __name__ == "__main__":
    main()