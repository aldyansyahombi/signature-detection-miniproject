"""Aturan sederhana penentuan SIGNATURE PRESENT / SIGNATURE ABSENT."""
from . import config


def decide(f):
    """
    Tanda tangan dianggap ADA bila SEMUA syarat terpenuhi:
      1. piksel foreground cukup banyak (bukan sekadar noise),
      2. rasio foreground terhadap luas ROI cukup (>= MIN_FG_RATIO),
      3. komponen terhubung terbesar cukup luas (goresan menyambung),
      4. foreground tidak berlebihan (rasio <= MAX_FG_RATIO); bila terlalu luas berarti
         threshold membelah tekstur kertas, bukan tinta,
      5. komponen terbesar membentang lebar (tanda tangan = goresan panjang,
         berbeda dari huruf cetak/bintik yang kecil dan terpisah).
    """
    ok = (f["fg_pixels"] >= config.MIN_FG_PIXELS
          and config.MIN_FG_RATIO <= f["fg_ratio"] <= config.MAX_FG_RATIO
          and f["largest_area"] >= config.MIN_LARGEST_COMP
          and f["largest_span_ratio"] >= config.MIN_SPAN_RATIO)
    return "SIGNATURE PRESENT" if ok else "SIGNATURE ABSENT"
