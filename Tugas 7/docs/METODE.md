# Penjelasan Metode & Analisis Enhancement (CER)

## 1. Pra-pemrosesan

| Tahap | Metode | Alasan |
|---|---|---|
| Koreksi orientasi | Jika portrait, coba rotasi 90° CW/CCW; pilih yang menghasilkan kata kunci OCR terbanyak (`ijazah`, `universitas`, `rektor`, …) | Hasil scan bisa terputar; ROI berbasis posisi butuh citra tegak |
| Standarisasi ukuran | Resize ke lebar 2000 px | Ambang & ukuran kernel konsisten di berbagai resolusi |
| Grayscale | `cv2.cvtColor` BGR→Gray | Informasi tinta/teks ada pada intensitas; mengurangi dimensi data |
| Enhancement global | Contrast stretching persentil 1–99 + CLAHE ringan (clip 1.5) | Menormalkan kontras/pencahayaan sebelum ROI dipotong |

## 2. Cabang nomor ijazah

1. **Area nomor** — crop berdasarkan rasio posisi, lalu dirapatkan secara vertikal dengan proyeksi horizontal.
2. **Enhancement ROI** — 10 kandidat dibandingkan (bagian 4).
3. **OCR** — Tesseract 5 (LSTM, `--oem 3 --psm 7`: satu baris teks).
4. **Parsing** — ambil token setelah `:`; koreksi karakter yang sering tertukar pada token numerik
   (`O→0`, `l/I→1`, `S→5`, `B→8`, …); buang karakter non-digit.

## 3. Cabang tanda tangan (Rektor)

1. **Normalisasi ROI** ke lebar 600 px.
2. **Enhancement**: estimasi latar dengan *morphological closing* (kernel 21×21) lalu pembagian
   `gray / background` — ini menghilangkan pola guilloche pada kertas sehingga tinta tampak gelap pada latar ≈255.
3. **Gerbang kontras**: `p90 − p1` harus ≥ 60. ROI kosong hanya berisi tekstur kertas (kontras kecil) → ABSENT.
4. **Thresholding**: Otsu (global) **AND** adaptive Gaussian (lokal); piksel dianggap tinta hanya bila keduanya setuju
   (menekan false positive dari tekstur).
5. **Morfologi**: *opening* (2×2) membuang bintik; *closing* (5×5) menyambung goresan yang putus.
6. **Connected components**: buang komponen < 0,03% luas ROI, hitung `ink_ratio` dan komponen terbesar.
7. **Keputusan**: `PRESENT` jika kontras ≥ 60 **dan** ink_ratio ≥ 0,004 **dan** komponen terbesar ≥ 0,0025; selain itu `ABSENT`.

## 4. Analisis enhancement berdasarkan CER

**CER** = (S + D + I) / N, yaitu jarak Levenshtein antara hasil OCR dan nomor sebenarnya dibagi panjang nomor
sebenarnya (N = 15). Semakin kecil semakin baik.

**Data uji:** 12 citra dari satu ijazah (9 kondisi degradasi simulasi: HighQuality, LowContrast, Blurred, HighNoise,
LowResolution, Faded, ColorShift, JPEGArtifacts, Combined; + 3 varian tanpa tanda tangan yang blur/noise).
Nomor referensi: `571012022000056`.

**Hasil (mean CER, 12 citra):**

| Peringkat | Metode | Mean CER | Exact match |
|---|---|---|---|
| 1 | raw_gray (tanpa enhancement apa pun) | 0,0056 | 11/12 |
| 2 | global_only (hanya enhancement global) | 0,0222 | 11/12 |
| 3 | median_clahe | 0,0278 | 9/12 |
| 4 | stretch | 0,0833 | 11/12 |
| 5 | unsharp | 0,0889 | 10/12 |
| 6 | clahe | 0,0944 | 9/12 |
| 7 | otsu | 0,0944 | 9/12 |
| 8 | denoise_clahe_x2 | 0,1000 | 8/12 |
| 9 | clahe_otsu | 0,1056 | 7/12 |
| 10 | x2_clahe | 0,2500 | 9/12 |
| 11 | adaptive | 0,3000 | 5/12 |

Tabel lengkap per citra: `results/cer_summary.md`; grafik: `results/cer_chart.png`.

**Kesimpulan:**

- Pada data ini **tidak ada metode enhancement yang memperbaiki hasil OCR**; baseline tanpa enhancement
  (`raw_gray`) memiliki CER terendah. Tesseract (LSTM) sudah melakukan normalisasi dan binarisasi internal,
  sehingga pemrosesan tambahan lebih sering menambah artefak daripada memperbaiki.
- Di antara metode enhancement yang benar-benar mengubah citra, yang paling aman adalah **enhancement global
  ringan (`global_only`)** dan **`median_clahe`** (denoise median + CLAHE). Karena itu `global_only` dipakai sebagai
  default pipeline.
- **Binarisasi keras (`adaptive`, `otsu`, `clahe_otsu`) dan `x2_clahe` paling buruk.** `adaptive` dan `x2_clahe`
  runtuh pada citra ber-noise tinggi (CER 1,00) karena noise ikut diperkuat/dibinarisasi menjadi bintik teks.
- Perbedaan ranking sebagian besar berasal dari citra `09_Combined` (blur + kontras rendah + noise + JPEG),
  satu-satunya kondisi yang benar-benar menyulitkan. Pada 8 dari 12 citra hampir semua metode CER = 0.

**Keterbatasan (penting):** hanya satu ijazah dan degradasi bersifat simulasi, sehingga kesimpulan ini belum
tentu berlaku pada ijazah asli yang bervariasi (scan miring, tinta pudar, stempel menimpa teks). Untuk kesimpulan
yang lebih kuat, tambahkan citra ijazah nyata ke `data/samples/` + `data/ground_truth.csv` lalu jalankan
`python scripts/evaluate.py`.

## 5. Hasil deteksi tanda tangan

Akurasi 12/12 pada set uji (9 citra dengan tanda tangan = PRESENT, 3 citra yang tanda tangannya dihapus = ABSENT).
Pada citra ABSENT, kontras ROI hanya 6–23 dan `ink_ratio` ≈ 0; pada PRESENT, kontras ≥ 206 dan `ink_ratio` ≈ 0,04.
Margin yang lebar ini menunjukkan ambang aman untuk set uji ini, tetapi ambang belum diuji pada ijazah lain
atau pada coretan/noda yang menyerupai tanda tangan (mis. stempel yang masuk ROI).
Detail: `results/signature_eval.md`.
