# Ijazah Verifier — OCR Nomor Ijazah + Deteksi Tanda Tangan

Prototype verifikasi citra ijazah menggunakan pengolahan citra klasik (OpenCV) dan Tesseract OCR.

**Input:** `ijazah_001.jpg` → **Output:**
```
Nomor Ijazah : 571012022000056
Tanda Tangan : PRESENT
```

## Pipeline

```
Citra Ijazah → (koreksi orientasi) → Grayscale → Image Enhancement (global)
   ├─ Area Nomor → Enhancement ROI → OCR (Tesseract) → Nomor Ijazah
   └─ Area Tanda Tangan → Thresholding (Otsu + Adaptive) → Morfologi → Signature Detection
                                   ↓
                            Hasil Verifikasi
```

Penjelasan metode lengkap + analisis CER: [`docs/METODE.md`](docs/METODE.md)

## Persyaratan

- Python 3.9+
- Tesseract OCR terpasang di sistem

| OS | Instalasi Tesseract |
|---|---|
| Ubuntu/Debian | `sudo apt install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | Unduh installer UB Mannheim, lalu tambahkan folder instalasi ke `PATH` |

## Instalasi

```bash
git clone <URL-REPO-ANDA>
cd ijazah-verifier
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cara menjalankan

**1. Verifikasi satu ijazah**
```bash
python verify.py data/samples/01_HighQuality_Enhanced.jpg
```

Opsi tambahan:
```bash
python verify.py ijazah_001.jpg --verbose              # tampilkan detail (OCR mentah, metrik tanda tangan)
python verify.py ijazah_001.jpg --debug results/debug  # simpan citra tiap tahap pipeline
python verify.py ijazah_001.jpg --enhancement clahe    # pilih metode enhancement ROI nomor
```

**2. Buat dataset uji (opsional)** — membuat 9 variasi degradasi + 3 citra tanpa tanda tangan dari 1 citra asli
```bash
python scripts/make_dataset.py data/samples/01_HighQuality_Enhanced.jpg
```

**3. Evaluasi CER + akurasi tanda tangan**
```bash
python scripts/evaluate.py
```
Hasil tersimpan di folder `results/`:
`cer_summary.md`, `cer_per_image.csv`, `cer_chart.png`, `signature_eval.md`.

## Struktur repo

```
ijazah-verifier/
├── verify.py               # CLI utama
├── ijazah/
│   ├── config.py           # ROI relatif & ambang deteksi
│   ├── preprocess.py       # orientasi, grayscale, enhancement global
│   ├── enhance.py          # 10 kandidat metode enhancement ROI
│   ├── ocr.py              # crop ROI nomor, Tesseract, parsing
│   ├── signature.py        # threshold + morfologi + keputusan
│   ├── metrics.py          # Levenshtein & CER
│   └── pipeline.py         # perakit pipeline
├── scripts/                # make_dataset.py, evaluate.py
├── data/                   # samples/ + ground_truth.csv
├── results/                # keluaran evaluasi
└── docs/METODE.md          # penjelasan metode & analisis CER
```

## Menggunakan ijazah lain

1. Letakkan citra di `data/samples/` dan tambahkan baris di `data/ground_truth.csv`.
2. ROI memakai **rasio posisi** (template tetap). Jika layout ijazah berbeda, sesuaikan `ROI` di `ijazah/config.py`
   (`nomor` = baris nomor ijazah, `rektor` = area tanda tangan). Gunakan `--debug` untuk memeriksa hasil crop.
3. Ambang tanda tangan (`SIG_*`) juga ada di `config.py`.

## Batasan

- ROI berbasis template, bukan deteksi layout otomatis.
- Nomor yang dibaca adalah baris "Nomor ijazah"; kode lain di ijazah (mis. `NC. 21- 006359`) tidak dibaca.
- Tanda tangan yang diperiksa adalah tanda tangan Rektor; sistem hanya menilai **ada/tidaknya** goresan tinta,
  bukan keaslian tanda tangan.
- Dataset evaluasi berasal dari **satu** ijazah dengan degradasi simulasi.
