"""Pemuatan citra, rotasi, cropping ROI, dan konversi grayscale."""
import cv2

from . import config


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Tidak bisa membaca citra: {path}")
    if config.ROTATE_CLOCKWISE_90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def crop_roi(img, roi):
    """Crop menggunakan koordinat relatif (x0, y0, x1, y1)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].copy()


def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def smooth(gray):
    k = config.BLUR_KSIZE
    return cv2.GaussianBlur(gray, (k, k), 0)
