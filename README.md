# Deteksi Keberadaan Tanda Tangan pada Ijazah (Miniproject Pengolahan Citra)

Program Python (OpenCV) yang menentukan apakah sebuah area pada citra ijazah mengandung tanda tangan:
**`SIGNATURE PRESENT`** atau **`SIGNATURE ABSENT`**.

| | |
|---|---|
| Nama | La Ode Muhammad Aldyansyah Ombi |
| NIM | F1G124036 |
| Kelas | A |

## Pipeline

1. **Preprocessing** – rotasi citra (scan tersimpan terputar 90°), crop ROI tanda tangan (Rektor & Dekan), konversi ke *grayscale*, Gaussian blur.
2. **Thresholding** – tiga metode dibandingkan: *global threshold* (T tetap = 110), *Otsu*, dan *adaptive Gaussian threshold*.
3. **Morphological operation** – *opening* (3×3, hapus bintik noise) lalu *closing* (7×7, sambung goresan terputus).
4. **Ekstraksi fitur area** – jumlah piksel foreground, rasio foreground, jumlah komponen terhubung, luas & lebar komponen terbesar.
5. **Aturan keputusan** – `SIGNATURE PRESENT` jika semua syarat terpenuhi (lihat `src/config.py` & `src/decision.py`):
   - `fg_pixels >= 1500`
   - `0.003 <= fg_ratio <= 0.25`
   - `largest_area >= 1200` piksel
   - `largest_span_ratio >= 0.35` (komponen terbesar membentang ≥ 35% lebar ROI)
6. **Pengujian** – 9 citra (berbagai degradasi) × 6 ROI = 54 pengujian (2 ROI bertanda tangan, 4 ROI tanpa tanda tangan).

## Struktur Repository

```
signature-detection-miniproject/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                  # entry point
├── src/
│   ├── __init__.py
│   ├── config.py            # ROI, parameter threshold/morfologi/aturan
│   ├── preprocessing.py     # load, rotasi, crop, grayscale
│   ├── segmentation.py      # global/Otsu/adaptive + morfologi
│   ├── features.py          # karakteristik area foreground
│   └── decision.py          # aturan PRESENT / ABSENT
├── data/
│   └── images/              # letakkan 9 citra .jpg di sini
├── results/                 # keluaran program (CSV + gambar)
└── docs/
    └── Laporan_Miniproject_Deteksi_Tanda_Tangan.docx
```

## Cara Menjalankan

**Prasyarat:** Python 3.9+.

```bash
# 1. Clone repository
git clone https://github.com/aldyansyahombi/signature-detection-miniproject.git
cd signature-detection-miniproject

# 2. (Opsional) buat virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependensi
pip install -r requirements.txt

# 4. Letakkan citra uji (.jpg/.png) di folder data/images/

# 5. Jalankan
python main.py
```

Opsi:

```bash
python main.py --data-dir data/images --out results --method adaptive
# --method : global | otsu | adaptive  (metode yang ditampilkan di tabel per-ROI)
```

## Keluaran (folder `results/`)

| File | Isi |
|---|---|
| `hasil_fitur.csv` | Fitur area & keputusan untuk setiap citra × ROI × metode |
| `ringkasan_akurasi.csv` | TP/TN/FP/FN dan akurasi per metode threshold |
| `pipeline_<nama_citra>.png` | Tahapan crop → grayscale → threshold → morfologi |
| `sensitivitas_threshold.png/.csv` | Efek threshold terlalu rendah / terlalu tinggi |

## Ringkasan Hasil

| Metode | TP | TN | FP | FN | Akurasi |
|---|---|---|---|---|---|
| Global (T=110) | 8 | 36 | 0 | 10 | 81,5% |
| Otsu | 18 | 36 | 0 | 0 | 100% |
| Adaptive Gaussian | 18 | 36 | 0 | 0 | 100% |

Global threshold gagal pada citra kontras rendah/blur dan memutus goresan tipis (tanda tangan Dekan).
Otsu hanya benar karena ada batas atas `MAX_FG_RATIO`; tanpa batas itu Otsu menganggap tekstur kertas pada ROI kosong sebagai foreground.
Analisis lengkap ada di `docs/Laporan_Miniproject_Deteksi_Tanda_Tangan.docx`.

## Catatan

- Sampel "tanpa tanda tangan" adalah area kertas kosong dari citra yang sama, termasuk 1 ROI berisi bintik tinta kecil (`neg_noda`) dan 1 ROI berisi teks cetak (`neg_teks`) sebagai *hard negative*.
- Seluruh 9 citra berasal dari satu ijazah yang sama (hanya berbeda degradasi), sehingga hasil menggambarkan ketahanan terhadap degradasi, bukan generalisasi ke ijazah lain. ROI memakai koordinat relatif dan perlu disesuaikan di `src/config.py` bila tata letak dokumen berbeda.
