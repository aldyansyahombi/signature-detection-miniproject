#!/usr/bin/env python3
"""Buat dataset uji dari 1 citra ijazah asli: 9 kondisi degradasi + 3 varian 'tanpa tanda tangan'.
Degradasi bersifat SIMULASI. Jika punya citra ijazah lain, taruh di data/samples/ dan
tambahkan barisnya di data/ground_truth.csv.
Pakai: python scripts/make_dataset.py data/samples/01_HighQuality_Enhanced.jpg
"""
import csv
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ijazah import config, preprocess  # noqa: E402

GT_NUMBER = "571012022000056"
rng = np.random.default_rng(42)


def low_contrast(im):  return np.clip(128 + (im.astype(np.float32) - 128) * 0.35, 0, 255).astype(np.uint8)
def blurred(im):       return cv2.GaussianBlur(im, (0, 0), 3.5)
def high_noise(im):    return np.clip(im.astype(np.float32) + rng.normal(0, 28, im.shape), 0, 255).astype(np.uint8)
def low_res(im):
    s = cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return cv2.resize(s, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
def faded(im):         return np.clip(im.astype(np.float32) * 0.55 + 110, 0, 255).astype(np.uint8)
def color_shift(im):
    f = im.astype(np.float32) * np.array([1.25, 0.9, 0.7])  # BGR: condong kebiruan/kemerahan
    return np.clip(f, 0, 255).astype(np.uint8)
def jpeg_art(im):
    ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 8])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)
def combined(im):
    return jpeg_art(high_noise(cv2.GaussianBlur(low_contrast(im), (0, 0), 2.0)))


DEGRADATIONS = {
    "HighQuality": lambda im: im, "LowContrast": low_contrast, "Blurred": blurred,
    "HighNoise": high_noise, "LowResolution": low_res, "Faded": faded,
    "ColorShift": color_shift, "JPEGArtifacts": jpeg_art, "Combined": combined,
}


def remove_signature(bgr_raw):
    """Hapus tanda tangan Rektor: tempel patch kertas kosong (tekstur latar tetap ada)."""
    up, rot = preprocess.auto_orient(bgr_raw)
    h, w = up.shape[:2]
    x0, x1, y0, y1 = config.ROI["rektor"]
    X0, X1, Y0, Y1 = int(x0 * w), int(x1 * w), int(y0 * h), int(y1 * h)
    px0, py0 = int(0.05 * w), int(0.02 * h)
    patch = up[py0:py0 + (Y1 - Y0), px0:px0 + (X1 - X0)]
    out = up.copy()
    out[Y0:Y1, X0:X1] = patch
    back = {"90cw": cv2.ROTATE_90_COUNTERCLOCKWISE, "90ccw": cv2.ROTATE_90_CLOCKWISE,
            "180": cv2.ROTATE_180, "0": None}[rot]
    return out if back is None else cv2.rotate(out, back)


def main(src):
    out_dir = "data/samples"
    os.makedirs(out_dir, exist_ok=True)
    raw = cv2.imread(src)
    rows = []
    for i, (name, fn) in enumerate(DEGRADATIONS.items(), 1):
        fname = f"{i:02d}_{name}.jpg"
        cv2.imwrite(f"{out_dir}/{fname}", fn(raw), [cv2.IMWRITE_JPEG_QUALITY, 92 if name != "JPEGArtifacts" else 95])
        rows.append((fname, GT_NUMBER, "PRESENT"))
    blank = remove_signature(raw)
    for tag, fn in (("HighQuality", lambda im: im), ("Blurred", blurred), ("HighNoise", high_noise)):
        fname = f"NoSig_{tag}.jpg"
        cv2.imwrite(f"{out_dir}/{fname}", fn(blank), [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append((fname, GT_NUMBER, "ABSENT"))
    with open("data/ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "nomor_ijazah", "tanda_tangan"])
        w.writerows(rows)
    print(f"{len(rows)} citra dibuat di {out_dir}, label di data/ground_truth.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/samples/01_HighQuality_Enhanced.jpg")
