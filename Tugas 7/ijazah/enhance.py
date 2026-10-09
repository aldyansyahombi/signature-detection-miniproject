"""Kandidat metode enhancement untuk ROI nomor ijazah (dibandingkan lewat CER)."""
import cv2
import numpy as np


def _clahe(g, clip=2.0):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4)).apply(g)


def _unsharp(g, amount=1.5, sigma=1.5):
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.addWeighted(g, 1 + amount, blur, -amount, 0)


def _x2(g):
    return cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def none(g):               return g
def clahe(g):              return _clahe(g)
def stretch(g):
    lo, hi = np.percentile(g, (2, 98))
    return np.clip((g.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
def unsharp(g):            return _unsharp(g)
def median_clahe(g):       return _clahe(cv2.medianBlur(g, 3))
def denoise_clahe_x2(g):   return _unsharp(_clahe(cv2.fastNlMeansDenoising(_x2(g), None, 10, 7, 21)), 1.0, 1.2)
def otsu(g):               return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def adaptive(g):           return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
def clahe_otsu(g):         return cv2.threshold(_clahe(cv2.GaussianBlur(g, (3, 3), 0)), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def x2_clahe(g):           return _clahe(_x2(g))


METHODS = {
    "global_only": none,
    "stretch": stretch,
    "clahe": clahe,
    "unsharp": unsharp,
    "median_clahe": median_clahe,
    "x2_clahe": x2_clahe,
    "denoise_clahe_x2": denoise_clahe_x2,
    "otsu": otsu,
    "adaptive": adaptive,
    "clahe_otsu": clahe_otsu,
}
