"""Deteksi tanda tangan: grayscale -> enhancement -> thresholding -> morfologi -> analisis komponen."""
import cv2
import numpy as np

from .config import (ROI, SIG_WIDTH, SIG_MIN_CONTRAST, SIG_MIN_INK_RATIO, SIG_MIN_COMP_AREA)
from .ocr import crop_ratio


def get_signature_roi(gray, name="rektor"):
    return crop_ratio(gray, ROI[name])


def detect_signature(roi_gray, return_debug=False):
    # 1. normalisasi ukuran supaya ambang tidak bergantung resolusi
    s = SIG_WIDTH / roi_gray.shape[1]
    g = cv2.resize(roi_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # 2. enhancement: hilangkan pola latar (guilloche) dengan estimasi background lalu normalisasi
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    norm = cv2.divide(g, bg, scale=255)  # tinta = gelap, kertas ~255

    # 3. gerbang kontras: ROI kosong hanya berisi tekstur kertas -> kontras kecil
    contrast = float(np.percentile(norm, 90) - np.percentile(norm, 1))

    # 4. thresholding: Otsu (global) AND adaptive (lokal) -> keduanya harus setuju
    _, t_otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t_adp = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 12)
    th = cv2.bitwise_and(t_otsu, t_adp)

    # 5. morfologi: opening buang bintik, closing sambung goresan putus
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

    # 6. analisis connected component
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    area_total = m.shape[0] * m.shape[1]
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    keep = areas >= 0.0003 * area_total  # buang komponen sangat kecil
    ink_ratio = float(areas[keep].sum() / area_total) if keep.any() else 0.0
    largest = float(areas.max() / area_total) if n > 1 else 0.0

    present = (contrast >= SIG_MIN_CONTRAST and ink_ratio >= SIG_MIN_INK_RATIO
               and largest >= SIG_MIN_COMP_AREA)
    info = {"contrast": round(contrast, 1), "ink_ratio": round(ink_ratio, 5),
            "largest_component": round(largest, 5), "decision": "PRESENT" if present else "ABSENT"}
    if return_debug:
        return present, info, {"norm": norm, "otsu": t_otsu, "adaptive": t_adp, "morph": m}
    return present, info
