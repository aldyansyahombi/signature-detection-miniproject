import cv2
import numpy as np
import pytesseract

from .config import WORK_WIDTH

VOCAB = ["universitas", "indonesia", "ijazah", "nomor", "memberikan", "kepada",
         "dekan", "rektor", "program", "studi", "fakultas", "magister", "sarjana"]


def _score_text(img_gray):
    small = img_gray
    if small.shape[1] > 1200:
        s = 1200 / small.shape[1]
        small = cv2.resize(small, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    txt = pytesseract.image_to_string(small, config="--oem 3 --psm 11").lower()
    return sum(1 for v in VOCAB if v in txt)


def auto_orient(bgr):
    """Putar citra agar tegak (landscape). Pilih kandidat rotasi dengan skor kata kunci OCR tertinggi."""
    h, w = bgr.shape[:2]
    if h > w:
        cands = {"90cw": cv2.ROTATE_90_CLOCKWISE, "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}
    else:
        cands = {"0": None, "180": cv2.ROTATE_180}
    best, best_s, best_name = bgr, -1, ""
    for name, code in cands.items():
        im = bgr if code is None else cv2.rotate(bgr, code)
        s = _score_text(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        if s > best_s:
            best, best_s, best_name = im, s, name
    return best, best_name


def standardize(bgr):
    """Samakan resolusi kerja (lebar WORK_WIDTH)."""
    h, w = bgr.shape[:2]
    s = WORK_WIDTH / w
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(bgr, None, fx=s, fy=s, interpolation=interp)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def global_enhance(gray):
    """Image Enhancement global (ringan): contrast stretching persentil 1-99 + CLAHE lembut."""
    lo, hi = np.percentile(gray, (1, 99))
    st = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(st)
