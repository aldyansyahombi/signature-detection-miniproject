"""Thresholding (global, Otsu, adaptive) dan operasi morfologi."""
import cv2

from . import config
from .preprocessing import smooth


def global_threshold(gray, t=None):
    """Threshold tetap. Tinta (gelap) -> foreground putih (255)."""
    t = config.GLOBAL_T if t is None else t
    _, b = cv2.threshold(smooth(gray), t, 255, cv2.THRESH_BINARY_INV)
    return b, float(t)


def otsu_threshold(gray):
    """Otsu: threshold dipilih otomatis dari histogram (memaksimalkan varians antar-kelas)."""
    t, b = cv2.threshold(smooth(gray), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b, float(t)


def adaptive_threshold(gray):
    """Adaptive Gaussian: threshold dihitung per jendela lokal."""
    b = cv2.adaptiveThreshold(smooth(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, config.ADAPTIVE_BLOCK, config.ADAPTIVE_C)
    return b, float("nan")


def morphology(binary):
    """Opening (hapus noise kecil) lalu closing (sambung goresan putus)."""
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.OPEN_KSIZE, config.OPEN_KSIZE))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.CLOSE_KSIZE, config.CLOSE_KSIZE))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ko)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kc)


METHODS = {
    "global": global_threshold,
    "otsu": otsu_threshold,
    "adaptive": adaptive_threshold,
}
