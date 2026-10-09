#!/usr/bin/env python3
"""Evaluasi: (1) CER OCR per metode enhancement, (2) akurasi deteksi tanda tangan.
Pakai: python scripts/evaluate.py
Output: results/cer_per_image.csv, results/cer_summary.md, results/cer_chart.png, results/signature_eval.md
"""
import csv
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ijazah import enhance, ocr, preprocess, signature, pipeline  # noqa: E402
from ijazah.metrics import cer  # noqa: E402

os.makedirs("results", exist_ok=True)
rows = list(csv.DictReader(open("data/ground_truth.csv")))

# ---------- 1. CER per metode enhancement (hanya citra dengan nomor terbaca di label) ----------
names = ["raw_gray"] + list(enhance.METHODS)  # raw_gray = tanpa enhancement sama sekali
table = {}   # file -> {method: cer}
detail = []
for r in rows:
    f = r["filename"]
    bgr = cv2.imread(f"data/samples/{f}")
    up, _ = preprocess.auto_orient(bgr)
    gray = preprocess.to_gray(preprocess.standardize(up))
    enh = preprocess.global_enhance(gray)
    roi = ocr.get_number_roi(enh)
    roi_raw = ocr.get_number_roi(gray)
    table[f] = {}
    for m in names:
        inp = roi_raw if m == "raw_gray" else enhance.METHODS[m](roi)
        pred, raw = ocr.read_number(inp)
        table[f][m] = cer(pred, r["nomor_ijazah"])
        detail.append((f, m, pred, round(table[f][m], 4)))
    print(f, {k: round(v, 2) for k, v in table[f].items()}, flush=True)

with open("results/cer_per_image.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["filename", "method", "prediction", "CER"])
    w.writerows(detail)

mean = {m: float(np.mean([table[f][m] for f in table])) for m in names}
exact = {m: sum(table[f][m] == 0 for f in table) for m in names}
order = sorted(names, key=lambda m: (mean[m], -exact[m]))
with open("results/cer_summary.md", "w") as fh:
    fh.write(f"| Peringkat | Metode enhancement | Mean CER | Exact match (dari {len(table)}) |\n|---|---|---|---|\n")
    for i, m in enumerate(order, 1):
        fh.write(f"| {i} | {m} | {mean[m]:.4f} | {exact[m]} |\n")
    fh.write("\n\nCER per citra (baris) x metode (kolom):\n\n| citra | " + " | ".join(names) + " |\n|" + "---|" * (len(names) + 1) + "\n")
    for f in table:
        fh.write(f"| {f} | " + " | ".join(f"{table[f][m]:.2f}" for m in names) + " |\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.barh([m for m in order][::-1], [mean[m] for m in order][::-1], color="#3b6ea5")
    plt.xlabel("Mean CER (lebih kecil = lebih baik)")
    plt.title("Perbandingan metode enhancement berdasarkan CER")
    plt.tight_layout()
    plt.savefig("results/cer_chart.png", dpi=150)
except Exception as e:  # matplotlib opsional
    print("chart dilewati:", e)

# ---------- 2. Deteksi tanda tangan ----------
ok = 0
lines = ["| citra | label | prediksi | contrast | ink_ratio | largest_comp |\n|---|---|---|---|---|---|\n"]
for r in rows:
    res = pipeline.run(f"data/samples/{r['filename']}")
    i = res["signature_info"]
    ok += res["tanda_tangan"] == r["tanda_tangan"]
    lines.append(f"| {r['filename']} | {r['tanda_tangan']} | {res['tanda_tangan']} | {i['contrast']} | {i['ink_ratio']} | {i['largest_component']} |\n")
with open("results/signature_eval.md", "w") as fh:
    fh.write(f"Akurasi deteksi tanda tangan: {ok}/{len(rows)} = {ok/len(rows):.2%}\n\n")
    fh.writelines(lines)
print(f"\nBest enhancement: {order[0]} (mean CER {mean[order[0]]:.4f})")
print(f"Akurasi tanda tangan: {ok}/{len(rows)}")
