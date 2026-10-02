"""Ekstraksi karakteristik area foreground hasil segmentasi."""
import cv2
import numpy as np


def extract_features(binary):
    h, w = binary.shape[:2]
    fg = int(np.count_nonzero(binary))
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = stats[1:]  # buang label 0 (background)
    if len(comps) == 0:
        return dict(fg_pixels=0, fg_ratio=0.0, n_components=0,
                    largest_area=0, largest_span_ratio=0.0)
    big = comps[np.argmax(comps[:, cv2.CC_STAT_AREA])]
    return dict(
        fg_pixels=fg,
        fg_ratio=fg / float(h * w),
        n_components=int(len(comps)),
        largest_area=int(big[cv2.CC_STAT_AREA]),
        largest_span_ratio=float(big[cv2.CC_STAT_WIDTH]) / w,
    )
