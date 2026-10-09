import re
import cv2
import numpy as np
import pytesseract

from .config import ROI

_CONF = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1",
                       "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9"})


def crop_ratio(img, roi):
    h, w = img.shape[:2]
    x0, x1, y0, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def tight_rows(gray):
    """Rapatkan crop secara vertikal pada baris teks (proyeksi horizontal)."""
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    prof = (bw > 0).sum(axis=1).astype(float)
    if prof.max() == 0:
        return gray
    rows = np.where(prof > 0.15 * prof.max())[0]
    pad = int(0.35 * (rows[-1] - rows[0] + 1))
    a, b = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, gray.shape[0])
    return gray[a:b]


def get_number_roi(gray):
    return tight_rows(crop_ratio(gray, ROI["nomor"]))


def parse_number(text):
    """Ambil nomor ijazah: token setelah 'ijazah:' / token terpanjang bernuansa digit."""
    text = text.strip().replace("\n", " ")
    m = re.search(r"[:;]\s*(.+)$", text)
    cand = m.group(1) if m else text
    cand = cand.replace(" ", "")
    prefix = re.match(r"^([A-Za-z]{1,3}-)", cand)
    body = cand[len(prefix.group(1)):] if prefix else cand
    body = body.translate(_CONF)
    body = re.sub(r"[^0-9\-]", "", body)
    digits = max(re.findall(r"\d[\d\-]*", body) or [""], key=len)
    return (prefix.group(1).upper() if prefix else "") + digits


def read_number(roi_img):
    """OCR (Tesseract LSTM, psm 7 = satu baris teks)."""
    h = roi_img.shape[0]
    if h < 64:  # Tesseract butuh tinggi teks memadai
        s = 64 / h
        roi_img = cv2.resize(roi_img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    roi_img = cv2.copyMakeBorder(roi_img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    raw = pytesseract.image_to_string(roi_img, config="--oem 3 --psm 7").strip()
    return parse_number(raw), raw
