# MeclisAI - TBMM Tutanak RAG Başlangıç Veri Seti

Bu paket, yüklenen 60 adet TBMM Genel Kurul tutanağından çıkarılmış metinleri ve RAG için hazırlanmış chunk dosyasını içerir.

Veri kapsamı:
- TBMM Genel Kurul Tutanakları
- 28. Dönem, 3. Yasama Yılı
- Birleşim no: 55-114
- Toplam PDF: 60
- Toplam çıkarılan karakter: 22,647,367
- Toplam chunk: 19607

Dosyalar:
- data/processed/documents.jsonl: Her PDF için tam metin + sayfa metinleri
- data/processed/chunks.jsonl: RAG için parçalara bölünmüş metinler
- data/processed/document_stats.csv: PDF bazlı sayfa/kelime/chunk istatistikleri
- src/extract_pdfs.py: PDF -> documents.jsonl
- src/create_chunks.py: documents.jsonl -> chunks.jsonl
- src/build_vector_db.py: chunks.jsonl -> FAISS index
- src/rag_query.py: FAISS üzerinden sorgu denemesi

Not:
108.pdf yalnızca 1 sayfa ve 1098 karakter metin çıkardı. Bu birleşimde oturum açılıp çalışmalar başlayamadığı için kısa olabilir; yine de elle kontrol etmek iyi olur.

Kurulum:
```bash
pip install -r requirements.txt
```

FAISS index oluşturma:
```bash
python src/build_vector_db.py
```

Sorgu deneme:
```bash
python src/rag_query.py
```
