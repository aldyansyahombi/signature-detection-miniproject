"""
Analisis Citra Ijazah: Grayscale, Histogram, dan Statistik Kualitas
-------------------------------------------------------------------
Kebutuhan : pip install opencv-python numpy matplotlib
Cara pakai: letakkan semua citra (png/jpg) di folder INPUT_DIR, lalu jalankan:
            python analisis_citra.py
Output    : folder OUTPUT_DIR berisi citra grayscale, panel histogram,
            dan results.json (statistik per citra)
"""
import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_DIR = "Tugas 3/data/images"      
OUTPUT_DIR = "Tugas 3/out"
EXT = (".png", ".jpg", ".jpeg")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def analisis_pencahayaan(gray, block=8):
    """Bagi citra jadi block x block petak, hitung rata-rata tiap petak.
    Std/rentang antar-petak yang besar => pencahayaan tidak merata."""
    h, w = gray.shape
    bh, bw = h // block, w // block
    means = []
    for by in range(block):
        for bx in range(block):
            patch = gray[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
            means.append(patch.mean())
    means = np.array(means)
    return means.std(), means.max() - means.min()


def proses(path, idx):
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Grayscale: 0.299 R + 0.587 G + 0.114 B
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    nz = np.nonzero(hist_gray)[0]
    total = gray.size

    illum_std, illum_range = analisis_pencahayaan(gray)

    stat = {
        "idx": idx,
        "file": os.path.basename(path),
        "mean": round(float(gray.mean()), 2),
        "std": round(float(gray.std()), 2),
        "min": int(gray.min()),
        "max": int(gray.max()),
        "pct_dark": round(float((gray < 60).sum() / total * 100), 2),
        "pct_bright": round(float((gray > 200).sum() / total * 100), 2),
        "range_used": [int(nz.min()), int(nz.max())],
        # Varian Laplacian: indikator tekstur/derau frekuensi tinggi
        "lap_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "illum_std": round(float(illum_std), 2),
        "illum_range": round(float(illum_range), 2),
    }

    # simpan citra grayscale
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"img{idx}_gray.png"), gray)

    # panel: asli, grayscale, histogram RGB, histogram grayscale
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].imshow(img_rgb)
    ax[0, 0].set_title(f"Citra Asli (RGB) - Img {idx}")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title(f"Citra Grayscale - Img {idx}")
    ax[0, 1].axis("off")

    for c, col in enumerate(("r", "g", "b")):
        h_c = cv2.calcHist([img_rgb], [c], None, [256], [0, 256])
        ax[1, 0].plot(h_c, color=col, alpha=0.8)
    ax[1, 0].set_title("Histogram RGB (asli)")
    ax[1, 0].set_xlim([0, 256])
    ax[1, 0].set_xlabel("Intensitas")
    ax[1, 0].set_ylabel("Jumlah piksel")

    ax[1, 1].plot(hist_gray, color="black")
    ax[1, 1].fill_between(range(256), hist_gray, color="gray", alpha=0.4)
    ax[1, 1].set_title("Histogram Grayscale")
    ax[1, 1].set_xlim([0, 256])
    ax[1, 1].set_xlabel("Intensitas")
    ax[1, 1].set_ylabel("Jumlah piksel")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"img{idx}_panel.png"), dpi=130)
    plt.close()
    return stat


def main():
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(EXT))
    hasil = []
    for i, f in enumerate(files, start=1):
        s = proses(os.path.join(INPUT_DIR, f), i)
        hasil.append(s)
        print(s)
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as fp:
        json.dump(hasil, fp, indent=2)


if __name__ == "__main__":
    main()
