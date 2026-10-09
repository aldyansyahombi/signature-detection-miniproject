"""Konfigurasi: ROI relatif (rasio terhadap citra yang sudah tegak) dan ambang."""

WORK_WIDTH = 2000  # semua citra distandarkan ke lebar ini (landscape)

# ROI = (x0, x1, y0, y1) sebagai rasio lebar/tinggi citra tegak.
# Layout ijazah dianggap tetap (template). Ubah jika templatenya berbeda.
ROI = {
    "nomor":  (0.03, 0.36, 0.89, 0.96),
    "rektor": (0.10, 0.42, 0.715, 0.82),
}

# Parameter deteksi tanda tangan (pada ROI yang dinormalisasi ke lebar 600 px)
SIG_WIDTH = 600
SIG_MIN_CONTRAST = 60      # selisih gray bg(p90) - ink(p1); di bawah ini -> ABSENT
SIG_MIN_INK_RATIO = 0.004  # rasio piksel foreground minimum
SIG_MIN_COMP_AREA = 0.0025 # luas komponen terbesar minimum (rasio terhadap ROI)

# Metode enhancement yang dipakai di pipeline akhir (diisi dari hasil evaluasi CER)
DEFAULT_OCR_ENHANCEMENT = "global_only"
