"""Deteksi keberadaan tanda tangan pada ijazah: SIGNATURE PRESENT / SIGNATURE ABSENT.

Contoh:
    python main.py
    python main.py --data-dir data/images --out results --method otsu
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.decision import decide
from src.features import extract_features
from src.preprocessing import crop_roi, load_image, to_gray
from src.segmentation import METHODS, global_threshold, morphology

ALL_ROI = {**{k: (v, True) for k, v in config.ROI_POSITIVE.items()},
           **{k: (v, False) for k, v in config.ROI_NEGATIVE.items()}}


def evaluate(images, out_dir):
    """Jalankan seluruh pipeline pada semua citra & ROI, simpan CSV fitur."""
    rows = []
    for path in images:
        img = load_image(path)
        for roi_name, (roi, truth) in ALL_ROI.items():
            gray = to_gray(crop_roi(img, roi))
            for m, fn in METHODS.items():
                binary, t = fn(gray)
                clean = morphology(binary)
                f = extract_features(clean)
                pred = decide(f)
                truth_lbl = "SIGNATURE PRESENT" if truth else "SIGNATURE ABSENT"
                rows.append(dict(image=path.name, roi=roi_name, method=m, threshold=t,
                                 **f, truth=truth_lbl, prediction=pred,
                                 correct=pred == truth_lbl))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hasil_fitur.csv", index=False)
    return df


def summarize(df):
    out = []
    for m, g in df.groupby("method", sort=False):
        tp = ((g.truth == "SIGNATURE PRESENT") & (g.prediction == "SIGNATURE PRESENT")).sum()
        tn = ((g.truth == "SIGNATURE ABSENT") & (g.prediction == "SIGNATURE ABSENT")).sum()
        fp = ((g.truth == "SIGNATURE ABSENT") & (g.prediction == "SIGNATURE PRESENT")).sum()
        fn = ((g.truth == "SIGNATURE PRESENT") & (g.prediction == "SIGNATURE ABSENT")).sum()
        out.append(dict(method=m, TP=tp, TN=tn, FP=fp, FN=fn, total=len(g),
                        accuracy=round((tp + tn) / len(g), 4)))
    return pd.DataFrame(out)


def pipeline_figure(path, out_dir):
    """Baris = ROI, kolom = tahap: crop, gray, global, Otsu, adaptive, adaptive+morfologi."""
    img = load_image(path)
    rois = ["ttd_rektor", "ttd_dekan", "neg_kiri"]
    cols = ["Crop (RGB)", "Grayscale", "Global", "Otsu", "Adaptive", "Adaptive + morfologi"]
    fig, ax = plt.subplots(len(rois), len(cols), figsize=(3.2 * len(cols), 2.2 * len(rois)))
    for i, r in enumerate(rois):
        crop = crop_roi(img, ALL_ROI[r][0])
        gray = to_gray(crop)
        g, tg = METHODS["global"](gray)
        o, to = METHODS["otsu"](gray)
        a, _ = METHODS["adaptive"](gray)
        panels = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), gray, g, o, a, morphology(a)]
        titles = [cols[0], cols[1], f"{cols[2]} (T={tg:.0f})", f"{cols[3]} (T={to:.0f})", cols[4], cols[5]]
        for j, (p, t) in enumerate(zip(panels, titles)):
            ax[i, j].imshow(p, cmap=None if p.ndim == 3 else "gray")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(t, fontsize=9)
            else:
                ax[i, j].set_title(t if j in (2, 3) else "", fontsize=9)
        ax[i, 0].set_ylabel(r, fontsize=9)
    fig.suptitle(path.stem, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"pipeline_{path.stem}.png", dpi=110)
    plt.close(fig)


def threshold_sensitivity(path, out_dir):
    """Efek threshold global terlalu rendah / terlalu tinggi."""
    img = load_image(path)
    ts = [30, 60, 90, 120, 150, 180, 210, 240]
    rois = ["ttd_dekan", "neg_kiri"]
    fig, ax = plt.subplots(len(rois) + 1, len(ts), figsize=(2.6 * len(ts), 7.2))
    ratios = {r: [] for r in rois}
    for i, r in enumerate(rois):
        gray = to_gray(crop_roi(img, ALL_ROI[r][0]))
        for j, t in enumerate(ts):
            b, _ = global_threshold(gray, t)
            ratios[r].append(np.count_nonzero(b) / b.size)
            ax[i, j].imshow(b, cmap="gray"); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_title(f"T={t}", fontsize=9)
        ax[i, 0].set_ylabel(r, fontsize=9)
    gs = ax[len(rois), 0].get_gridspec()
    for a in ax[len(rois), :]:
        a.remove()
    axl = fig.add_subplot(gs[len(rois), :])
    for r in rois:
        axl.plot(ts, [x * 100 for x in ratios[r]], marker="o", label=r)
    axl.axhspan(0, config.MIN_FG_RATIO * 100, color="grey", alpha=0.15, label="zona 'tidak ada ink'")
    axl.axhline(config.MAX_FG_RATIO * 100, color="red", ls="--", label="batas atas rasio")
    axl.set_xlabel("Nilai threshold global T"); axl.set_ylabel("Foreground (% luas ROI)")
    axl.legend(fontsize=8); axl.grid(alpha=0.3)
    fig.suptitle(f"Sensitivitas threshold global - {path.stem}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivitas_threshold.png", dpi=110)
    plt.close(fig)
    pd.DataFrame({"T": ts, **{f"fg_ratio_{r}": ratios[r] for r in rois}}).to_csv(
        out_dir / "sensitivitas_threshold.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description="Deteksi tanda tangan (SIGNATURE PRESENT/ABSENT)")
    ap.add_argument("--data-dir", default="data/images", help="folder citra .jpg/.png")
    ap.add_argument("--out", default="results", help="folder keluaran")
    ap.add_argument("--method", default=config.DECISION_METHOD, choices=list(METHODS),
                    help="metode threshold yang ditampilkan pada hasil per-ROI")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"Tidak ada citra di {data_dir}")

    df = evaluate(images, out_dir)
    summ = summarize(df)
    summ.to_csv(out_dir / "ringkasan_akurasi.csv", index=False)

    sel = df[df.method == args.method]
    print(f"\n=== Hasil per citra & ROI (metode: {args.method}) ===")
    print(sel[["image", "roi", "fg_pixels", "fg_ratio", "largest_area",
               "largest_span_ratio", "prediction", "correct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Perbandingan metode threshold ===")
    print(summ.to_string(index=False))

    for p in images:
        pipeline_figure(p, out_dir)
    threshold_sensitivity(images[0], out_dir)
    print(f"\nSelesai. Keluaran disimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
