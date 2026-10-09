import cv2
import numpy as np
from pathlib import Path


# ==============================
# FOLDER
# ==============================

INPUT_FOLDER = Path("citra")
OUTPUT_FOLDER = Path("hasil")

GRAY_FOLDER = OUTPUT_FOLDER / "grayscale"
MEAN_FOLDER = OUTPUT_FOLDER / "mean"
MEDIAN_FOLDER = OUTPUT_FOLDER / "median"
GAUSSIAN_FOLDER = OUTPUT_FOLDER / "gaussian"
SHARPENING_FOLDER = OUTPUT_FOLDER / "sharpening"

# Folder tambahan untuk hasil perbandingan
COMPARISON_FOLDER = OUTPUT_FOLDER / "perbandingan"


# ==============================
# MEMBUAT FOLDER JIKA BELUM ADA
# ==============================

GRAY_FOLDER.mkdir(parents=True, exist_ok=True)
MEAN_FOLDER.mkdir(parents=True, exist_ok=True)
MEDIAN_FOLDER.mkdir(parents=True, exist_ok=True)
GAUSSIAN_FOLDER.mkdir(parents=True, exist_ok=True)
SHARPENING_FOLDER.mkdir(parents=True, exist_ok=True)

COMPARISON_FOLDER.mkdir(parents=True, exist_ok=True)


# ==============================
# MENCARI SEMUA CITRA
# ==============================

extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

image_files = []

for ext in extensions:
    image_files.extend(INPUT_FOLDER.glob(ext))

image_files = sorted(image_files)


if not image_files:
    print("Tidak ada citra yang ditemukan di folder citra/")
    exit()


print("=" * 60)
print("ENHANCEMENT CITRA UNTUK OCR")
print("=" * 60)
print(f"Jumlah citra: {len(image_files)}")
print()


# ==============================
# PROSES SETIAP CITRA
# ==============================

for image_path in image_files:

    print(f"Memproses: {image_path.name}")


    # ==============================
    # MEMBACA CITRA
    # ==============================

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"  Gagal membaca: {image_path.name}")
        continue


    # ==============================
    # 1. GRAYSCALE
    # ==============================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ==============================
    # 2. MEAN FILTER
    # ==============================

    mean_filter = cv2.blur(
        gray,
        (3, 3)
    )


    # ==============================
    # 3. MEDIAN FILTER
    # ==============================

    median_filter = cv2.medianBlur(
        gray,
        3
    )


    # ==============================
    # 4. GAUSSIAN FILTER
    # ==============================

    gaussian_filter = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )


    # ==============================
    # 5. SHARPENING
    # ==============================

    kernel_sharpening = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpening = cv2.filter2D(
        gray,
        -1,
        kernel_sharpening
    )


    # ==============================
    # NAMA FILE OUTPUT
    # ==============================

    filename = image_path.stem + ".png"


    # ==============================
    # SIMPAN HASIL GRAYSCALE
    # ==============================

    cv2.imwrite(
        str(GRAY_FOLDER / filename),
        gray
    )


    # ==============================
    # SIMPAN HASIL MEAN FILTER
    # ==============================

    cv2.imwrite(
        str(MEAN_FOLDER / filename),
        mean_filter
    )


    # ==============================
    # SIMPAN HASIL MEDIAN FILTER
    # ==============================

    cv2.imwrite(
        str(MEDIAN_FOLDER / filename),
        median_filter
    )


    # ==============================
    # SIMPAN HASIL GAUSSIAN FILTER
    # ==============================

    cv2.imwrite(
        str(GAUSSIAN_FOLDER / filename),
        gaussian_filter
    )


    # ==============================
    # SIMPAN HASIL SHARPENING
    # ==============================

    cv2.imwrite(
        str(SHARPENING_FOLDER / filename),
        sharpening
    )


    # ==========================================================
    # 6. MEMBUAT HASIL PERBANDINGAN
    # ==========================================================

    # --------------------------------
    # Original
    # --------------------------------
    original_bgr = image.copy()


    # --------------------------------
    # Hasil filter diubah ke BGR
    # --------------------------------

    mean_bgr = cv2.cvtColor(
        mean_filter,
        cv2.COLOR_GRAY2BGR
    )

    median_bgr = cv2.cvtColor(
        median_filter,
        cv2.COLOR_GRAY2BGR
    )

    gaussian_bgr = cv2.cvtColor(
        gaussian_filter,
        cv2.COLOR_GRAY2BGR
    )

    sharpening_bgr = cv2.cvtColor(
        sharpening,
        cv2.COLOR_GRAY2BGR
    )


    # ==========================================================
    # UKURAN PANEL
    # ==========================================================

    panel_width = 500

    original_height = int(
        original_bgr.shape[0]
        * panel_width
        / original_bgr.shape[1]
    )

    filter_height = int(
        mean_bgr.shape[0]
        * panel_width
        / mean_bgr.shape[1]
    )


    # Resize Original
    original_bgr = cv2.resize(
        original_bgr,
        (panel_width, original_height)
    )


    # Resize hasil filter
    mean_bgr = cv2.resize(
        mean_bgr,
        (panel_width, filter_height)
    )

    median_bgr = cv2.resize(
        median_bgr,
        (panel_width, filter_height)
    )

    gaussian_bgr = cv2.resize(
        gaussian_bgr,
        (panel_width, filter_height)
    )

    sharpening_bgr = cv2.resize(
        sharpening_bgr,
        (panel_width, filter_height)
    )


    # ==========================================================
    # FUNGSI MEMBERI JUDUL
    # ==========================================================

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.75
    thickness = 2

    title_height = 40

    text_color = (255, 255, 255)


    def add_title(img, title):

        result = cv2.copyMakeBorder(
            img,
            title_height,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        text_size = cv2.getTextSize(
            title,
            font,
            font_scale,
            thickness
        )[0]

        text_x = (
            img.shape[1] - text_size[0]
        ) // 2

        text_y = 27

        cv2.putText(
            result,
            title,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

        return result


    # ==========================================================
    # TAMBAHKAN JUDUL
    # ==========================================================

    original_bgr = add_title(
        original_bgr,
        "ORIGINAL"
    )

    mean_bgr = add_title(
        mean_bgr,
        "MEAN FILTER"
    )

    median_bgr = add_title(
        median_bgr,
        "MEDIAN FILTER"
    )

    gaussian_bgr = add_title(
        gaussian_bgr,
        "GAUSSIAN FILTER"
    )

    sharpening_bgr = add_title(
        sharpening_bgr,
        "SHARPENING"
    )


    # ==========================================================
    # JARAK ANTAR PANEL
    # ==========================================================

    gap = 20

    # Warna putih untuk jarak
    horizontal_gap = np.ones(
        (original_bgr.shape[0], gap, 3),
        dtype=np.uint8
    ) * 255

    horizontal_gap_filter = np.ones(
        (mean_bgr.shape[0], gap, 3),
        dtype=np.uint8
    ) * 255


    # ==========================================================
    # BARIS 2
    # MEAN + MEDIAN
    # ==========================================================

    baris_kedua = cv2.hconcat([
        mean_bgr,
        horizontal_gap_filter,
        median_bgr
    ])


    # ==========================================================
    # BARIS 3
    # GAUSSIAN + SHARPENING
    # ==========================================================

    baris_ketiga = cv2.hconcat([
        gaussian_bgr,
        horizontal_gap_filter,
        sharpening_bgr
    ])


    # ==========================================================
    # ORIGINAL DI TENGAH ATAS
    # ==========================================================

    total_width = (
        panel_width * 2
        + gap
    )

    # Jarak kiri dan kanan untuk Original
    side_space = (
        total_width - original_bgr.shape[1]
    ) // 2

    left_space = np.ones(
        (
            original_bgr.shape[0],
            side_space,
            3
        ),
        dtype=np.uint8
    ) * 255

    right_space = np.ones(
        (
            original_bgr.shape[0],
            total_width
            - original_bgr.shape[1]
            - side_space,
            3
        ),
        dtype=np.uint8
    ) * 255


    baris_pertama = cv2.hconcat([
        left_space,
        original_bgr,
        right_space
    ])


    # ==========================================================
    # JARAK VERTIKAL ANTAR BARIS
    # ==========================================================

    vertical_gap = 25

    gap_atas = np.ones(
        (
            vertical_gap,
            total_width,
            3
        ),
        dtype=np.uint8
    ) * 255

    gap_bawah = np.ones(
        (
            vertical_gap,
            total_width,
            3
        ),
        dtype=np.uint8
    ) * 255


    # ==========================================================
    # GABUNGKAN SEMUA
    # ==========================================================

    comparison = cv2.vconcat([
        baris_pertama,
        gap_atas,
        baris_kedua,
        gap_bawah,
        baris_ketiga
    ])


    # ==========================================================
    # SIMPAN HASIL PERBANDINGAN
    # ==========================================================

    cv2.imwrite(
        str(COMPARISON_FOLDER / filename),
        comparison
    )


    # ==============================
    # INFORMASI TERMINAL
    # ==============================

    print("  ✓ Grayscale")
    print("  ✓ Mean Filter")
    print("  ✓ Median Filter")
    print("  ✓ Gaussian Filter")
    print("  ✓ Sharpening")
    print("  ✓ Perbandingan")
    print()


# ==============================
# SELESAI
# ==============================

print("=" * 60)
print("SELESAI")
print("=" * 60)

print("Hasil tersimpan di folder: hasil/")
print()

print("Folder hasil:")
print("  - grayscale/")
print("  - mean/")
print("  - median/")
print("  - gaussian/")
print("  - sharpening/")
print("  - perbandingan/")