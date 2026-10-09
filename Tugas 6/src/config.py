"""Konfigurasi miniproject: ROI, parameter threshold, morfologi, dan aturan keputusan."""

# --- Orientasi: citra ijazah hasil scan tersimpan portrait (terputar 90 derajat).
# Diputar searah jarum jam 90 derajat agar teks terbaca normal (landscape).
ROTATE_CLOCKWISE_90 = True

# --- ROI dalam koordinat relatif (x0, y0, x1, y1) terhadap citra landscape.
# Relatif (0-1) agar tetap valid pada citra beresolusi berbeda.
ROI_POSITIVE = {
    "ttd_rektor": (0.140, 0.715, 0.405, 0.820),
    "ttd_dekan": (0.625, 0.700, 0.875, 0.820),
}
# Area tanpa tanda tangan (kertas kosong). 'neg_noda' memuat bintik tinta kecil,
# 'neg_teks' memuat teks cetak -> negatif yang sulit (hard negative).
ROI_NEGATIVE = {
    "neg_kiri": (0.020, 0.120, 0.280, 0.240),
    "neg_kanan": (0.750, 0.200, 0.980, 0.320),
    "neg_noda": (0.900, 0.360, 0.995, 0.580),
    "neg_teks": (0.125, 0.820, 0.400, 0.865),
}

# --- Thresholding
GLOBAL_T = 110            # nilai threshold global tetap (0-255)
BLUR_KSIZE = 5            # Gaussian blur sebelum threshold (harus ganjil)
ADAPTIVE_BLOCK = 51       # ukuran jendela lokal adaptive threshold (ganjil)
ADAPTIVE_C = 15           # konstanta pengurang adaptive threshold

# --- Morphological operation
OPEN_KSIZE = 3            # opening: buang bintik/noise kecil
CLOSE_KSIZE = 7           # closing: sambung goresan yang terputus

# --- Aturan keputusan (dikalibrasi pada citra uji)
MIN_FG_PIXELS = 1500      # jumlah piksel foreground minimum
MIN_FG_RATIO = 0.003      # rasio foreground / luas ROI minimum
MIN_LARGEST_COMP = 1200   # luas komponen terhubung terbesar minimum (piksel)
MIN_SPAN_RATIO = 0.35     # lebar bounding box komponen terbesar / lebar ROI
MAX_FG_RATIO = 0.25       # batas atas: foreground terlalu luas = threshold membanjiri kertas/tekstur

DECISION_METHOD = "adaptive"  # metode threshold utama untuk keputusan: global | otsu | adaptive
