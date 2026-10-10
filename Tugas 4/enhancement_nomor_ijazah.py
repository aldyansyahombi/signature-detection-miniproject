"""
enhancement_nomor_ijazah.py
============================
Tugas: Peningkatan kualitas area "Nomor ijazah" pada citra agar karakter
lebih mudah dikenali, menggunakan tiga metode enhancement:
    1. Brightness Adjustment
    2. Contrast Stretching
    3. Histogram Equalization

Untuk tiap citra, ketiga metode dibandingkan secara VISUAL (citra + histogram)
dan secara OBJEKTIF menggunakan OCR (Tesseract) yang hasil teksnya
dibandingkan dengan ground truth nomor ijazah.

Requirement:
    pip install opencv-python-headless numpy matplotlib pytesseract
    sudo apt install tesseract-ocr

Struktur folder yang diasumsikan:
    ./uploads/   -> berisi 9 file citra (01_....jpg ... 09_....jpg)
    ./outputs/   -> hasil figure (.png) dan results.json akan disimpan di sini
"""

import os
import json
import difflib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytesseract

# ---------------------------------------------------------------------------
# 0. KONFIGURASI
# ---------------------------------------------------------------------------

UPLOAD_DIR = "uploads"     # folder berisi citra asli
OUTPUT_DIR = "outputs"     # folder hasil (figure & json)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "01_HighQuality_Enhanced.jpg",
    "02_LowContrast.jpg",
    "03_Blurred.jpg",
    "04_HighNoise.jpg",
    "05_LowResolution_Upsampled.jpg",
    "06_Faded_Underexposed.jpg",
    "07_ColorShift_WarmTint.jpg",
    "08_JPEGCompression_Artifacts.jpg",
    "09_CombinedDegradation.jpg",
]

# Ground truth teks pada area nomor ijazah, dipakai untuk menilai akurasi OCR
GROUND_TRUTH = "Nomor ijazah: 5710120220000056"


# ---------------------------------------------------------------------------
# 1. FUNGSI CROP ROI (Region of Interest)
# ---------------------------------------------------------------------------

def load_and_crop_roi(path: str) -> np.ndarray:
    """
    Membaca citra dan mengambil area "Nomor ijazah" (ROI), lalu
    mengonversinya ke grayscale.

    Catatan: pada dataset ini, tiap file yang diunggah SUDAH berupa hasil
    crop dari area nomor ijazah (strip vertikal sempit). Jika kamu bekerja
    dengan citra ijazah utuh (belum di-crop), ganti bagian ROI di bawah
    dengan koordinat bounding box area nomor ijazah, misalnya:

        roi = gray[y1:y2, x1:x2]

    di mana (x1,y1)-(x2,y2) adalah koordinat kotak area nomor ijazah.
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Tidak bisa membaca citra: {path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Contoh cropping manual (aktifkan & sesuaikan jika perlu) ---
    # h, w = gray.shape
    # roi = gray[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]
    # return roi

    return gray  # citra input sudah berupa ROI nomor ijazah


# ---------------------------------------------------------------------------
# 2. TIGA METODE ENHANCEMENT
# ---------------------------------------------------------------------------

def brightness_adjustment(gray: np.ndarray, target_mean: float = 170) -> np.ndarray:
    """
    Brightness adjustment sederhana: menggeser (offset) seluruh nilai
    piksel sehingga rata-rata intensitas citra mendekati target_mean.
    Ini TIDAK mengubah rentang/kontras, hanya menggeser kecerahan.
    """
    delta = target_mean - gray.mean()
    out = gray.astype(np.float32) + delta
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_stretching(gray: np.ndarray) -> np.ndarray:
    """
    Contrast stretching (linear normalization) berbasis persentil 2-98,
    supaya tidak terlalu sensitif terhadap outlier (piksel ekstrem).

    Rumus:
        out = (in - low) * 255 / (high - low)

    di mana `low` dan `high` adalah persentil ke-2 dan ke-98 dari
    distribusi intensitas citra.
    """
    low, high = np.percentile(gray, (2, 98))
    if high <= low:  # fallback jika citra nyaris konstan
        low, high = gray.min(), gray.max()
        if high <= low:
            return gray.copy()
    out = (gray.astype(np.float32) - low) * 255.0 / (high - low)
    return np.clip(out, 0, 255).astype(np.uint8)


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """
    Histogram equalization global menggunakan OpenCV. Meratakan
    distribusi intensitas piksel di seluruh rentang 0-255 berdasarkan
    CDF (cumulative distribution function) histogram citra.
    """
    return cv2.equalizeHist(gray)


# ---------------------------------------------------------------------------
# 3. VALIDASI OBJEKTIF: OCR + METRIK KONTRAS
# ---------------------------------------------------------------------------

def ocr_read(gray: np.ndarray) -> str:
    """
    Menjalankan OCR (Tesseract) pada citra grayscale.
    Karena teks pada ROI tercetak vertikal (rotasi 90 derajat), citra
    dirotasi dahulu agar teks menjadi horizontal sebelum di-OCR.
    """
    rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    # Upscale citra kecil supaya OCR lebih akurat
    h, w = rotated.shape
    if h < 100:
        scale = 100 / h
        rotated = cv2.resize(
            rotated, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # --psm 7 = treat the image as a single text line
    text = pytesseract.image_to_string(rotated, config="--psm 7")
    return text.strip()


def text_similarity(a: str, b: str) -> float:
    """Kemiripan string 0-1 antara hasil OCR dan ground truth."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def michelson_contrast(gray: np.ndarray) -> float:
    """
    Kontras Michelson = (Lmax - Lmin) / (Lmax + Lmin)
    Lmax = rata-rata intensitas piksel "terang" (persentil atas, latar)
    Lmin = rata-rata intensitas piksel "gelap" (persentil bawah, teks/tinta)
    Semakin besar nilainya, semakin jelas pemisahan teks dari latar.
    """
    fg = gray[gray < np.percentile(gray, 30)]   # piksel gelap (diduga teks)
    bg = gray[gray > np.percentile(gray, 70)]   # piksel terang (diduga latar)
    if len(fg) == 0 or len(bg) == 0:
        return 0.0
    l_max, l_min = bg.mean(), fg.mean()
    if l_max + l_min == 0:
        return 0.0
    return float((l_max - l_min) / (l_max + l_min))


# ---------------------------------------------------------------------------
# 4. VISUALISASI: citra + histogram, sebelum & sesudah, per metode
#    Layout 2x2 (bukan 4 kolom sejajar) supaya tiap panel cukup besar
#    dan mudah dibaca:
#        [Original]        [Brightness]
#        [hist Original]   [hist Brightness]
#        [Contrast Stretch][Hist. Equalization]
#        [hist CS]         [hist HE]
# ---------------------------------------------------------------------------

# Posisi tiap metode di grid 4 baris x 2 kolom: (baris_gambar, baris_histogram)
GRID_POSITION = {
    "Original":            {"img": (0, 0), "hist": (1, 0)},
    "Brightness":          {"img": (0, 1), "hist": (1, 1)},
    "Contrast Stretch":    {"img": (2, 0), "hist": (3, 0)},
    "Hist. Equalization":  {"img": (2, 1), "hist": (3, 1)},
}


def make_comparison_figure(fname: str, methods: dict, metrics: dict, out_path: str):
    fig = plt.figure(figsize=(10, 13))
    gs = gridspec.GridSpec(
        4, 2, height_ratios=[3, 1.1, 3, 1.1],
        hspace=0.45, wspace=0.25, top=0.93, bottom=0.04, left=0.08, right=0.96,
    )

    for method_name, img in methods.items():
        (r_img, c_img) = GRID_POSITION[method_name]["img"]
        (r_hist, c_hist) = GRID_POSITION[method_name]["hist"]
        d = metrics[method_name]

        ax_img = fig.add_subplot(gs[r_img, c_img])
        ax_img.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax_img.set_title(f"{method_name}\nOCR: '{d['ocr'][:34]}'",
                          fontsize=11, fontweight="bold")
        ax_img.axis("off")

        ax_hist = fig.add_subplot(gs[r_hist, c_hist])
        ax_hist.hist(img.ravel(), bins=256, range=(0, 255), color="steelblue")
        ax_hist.set_title(f"Histogram · sim={d['sim']:.2f} · std={d['std']:.1f}",
                           fontsize=9.5)
        ax_hist.set_xlim(0, 255)
        ax_hist.tick_params(labelsize=8)

    fig.suptitle(fname, fontsize=14, fontweight="bold")
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def make_summary_chart(results: list, out_path: str):
    """Bar chart rata-rata akurasi OCR (kemiripan teks) per metode, 9 citra."""
    method_names = ["Original", "Brightness", "Contrast Stretch", "Hist. Equalization"]
    means = [np.mean([r[m]["sim"] for r in results]) for m in method_names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#94a3b8", "#60a5fa", "#34d399", "#f87171"]
    bars = ax.bar(range(len(method_names)), means, color=colors)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rata-rata kemiripan teks OCR terhadap ground truth")
    ax.set_title("Rata-rata akurasi OCR (9 citra) per metode enhancement")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    all_results = []

    for fname in FILES:
        path = os.path.join(UPLOAD_DIR, fname)
        roi = load_and_crop_roi(path)

        # --- terapkan tiga metode enhancement ---
        methods = {
            "Original": roi,
            "Brightness": brightness_adjustment(roi),
            "Contrast Stretch": contrast_stretching(roi),
            "Hist. Equalization": histogram_equalization(roi),
        }

        # --- validasi tiap hasil dengan OCR + metrik kontras ---
        metrics = {}
        for method_name, img in methods.items():
            ocr_text = ocr_read(img)
            metrics[method_name] = {
                "ocr": ocr_text,
                "sim": text_similarity(ocr_text, GROUND_TRUTH),
                "std": float(img.std()),
                "michelson": michelson_contrast(img),
            }

        row = {"file": fname, **metrics}
        all_results.append(row)

        # --- simpan figure perbandingan (citra + histogram) ---
        out_fig = os.path.join(OUTPUT_DIR, f"compare_{fname.replace('.jpg', '.png')}")
        make_comparison_figure(fname, methods, metrics, out_fig)
        print(f"[OK] {fname} -> {out_fig}")

    # --- simpan ringkasan hasil (JSON) & chart ringkasan ---
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    make_summary_chart(all_results, os.path.join(OUTPUT_DIR, "summary_ocr_accuracy.png"))

    # --- cetak tabel ringkas ke terminal ---
    method_names = ["Original", "Brightness", "Contrast Stretch", "Hist. Equalization"]
    print("\n=== RINGKASAN ===")
    print(f"{'File':35s} {'Method':18s} {'OCR text':32s} {'Sim':>5s} {'Std':>6s} {'Michelson':>9s}")
    for row in all_results:
        for m in method_names:
            d = row[m]
            print(f"{row['file']:35s} {m:18s} {d['ocr'][:32]:32s} "
                  f"{d['sim']:.2f} {d['std']:6.1f} {d['michelson']:9.3f}")

    print("\n=== RATA-RATA PER METODE ===")
    for m in method_names:
        sims = [r[m]["sim"] for r in all_results]
        stds = [r[m]["std"] for r in all_results]
        michs = [r[m]["michelson"] for r in all_results]
        print(f"{m:20s} sim={np.mean(sims):.3f}  std={np.mean(stds):.1f}  "
              f"michelson={np.mean(michs):.3f}")


if __name__ == "__main__":
    main()
