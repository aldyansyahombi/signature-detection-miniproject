import cv2
import pytesseract

from pathlib import Path


# ==========================================================
# PENGATURAN TESSERACT
# ==========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==========================================================
# FOLDER
# ==========================================================

INPUT_FOLDER = Path("citra")

OUTPUT_FOLDER = Path("hasil")

MEAN_FOLDER = OUTPUT_FOLDER / "mean"

MEDIAN_FOLDER = OUTPUT_FOLDER / "median"

GAUSSIAN_FOLDER = OUTPUT_FOLDER / "gaussian"

SHARPENING_FOLDER = OUTPUT_FOLDER / "sharpening"

OCR_RESULT_FILE = OUTPUT_FOLDER / "hasil_ocr.txt"


# ==========================================================
# GROUND TRUTH
# ==========================================================

GROUND_TRUTH = "Nomor ijazah: 571012022000056"


# ==========================================================
# CEK TESSERACT
# ==========================================================

try:

    tesseract_version = (
        pytesseract.get_tesseract_version()
    )

    print("=" * 70)

    print(
        "TESSERACT BERHASIL DITEMUKAN"
    )

    print("=" * 70)

    print(
        f"Versi Tesseract: {tesseract_version}"
    )

    print()


except Exception as error:

    print("=" * 70)

    print(
        "TESSERACT OCR TIDAK DITEMUKAN"
    )

    print("=" * 70)

    print()

    print(
        "Periksa apakah file berikut benar-benar ada:"
    )

    print()

    print(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    print()

    print(
        "Detail error:"
    )

    print(error)

    print()

    exit()


# ==========================================================
# MENCARI SEMUA CITRA
# ==========================================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp"
]


image_files = []


for ext in extensions:

    image_files.extend(
        INPUT_FOLDER.glob(ext)
    )


image_files = sorted(
    image_files
)


if not image_files:

    print("=" * 70)

    print(
        "TIDAK ADA CITRA"
    )

    print("=" * 70)

    print()

    print(
        "Tidak ada citra yang ditemukan di folder citra/"
    )

    print()

    exit()


# ==========================================================
# FUNGSI OCR
# ==========================================================

def lakukan_ocr(image):

    # ------------------------------------------------------
    # Jika citra masih berwarna, ubah ke grayscale
    # ------------------------------------------------------

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image


    # ------------------------------------------------------
    # Perbesar citra
    # ------------------------------------------------------

    scale = 2

    enlarged = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


    # ------------------------------------------------------
    # Konfigurasi OCR
    # ------------------------------------------------------
    # PSM 7 = satu baris teks
    #
    # Tidak menggunakan whitelist angka saja
    # agar huruf, angka, spasi, dan ":" dapat terbaca.
    # ------------------------------------------------------

    config = (
        "--oem 3 "
        "--psm 7"
    )


    # ------------------------------------------------------
    # Jalankan OCR
    # ------------------------------------------------------

    text = pytesseract.image_to_string(
        enlarged,
        config=config
    )


    # ------------------------------------------------------
    # Bersihkan hasil OCR
    # ------------------------------------------------------

    text = text.strip()

    text = " ".join(
        text.split()
    )


    return text


# ==========================================================
# HASIL OCR
# ==========================================================

hasil_ocr = []


# ==========================================================
# PROSES SETIAP CITRA
# ==========================================================

print("=" * 70)

print(
    "OCR NOMOR IJAZAH"
)

print("=" * 70)

print(
    f"Jumlah citra: {len(image_files)}"
)

print()

print(
    f"Ground Truth: {GROUND_TRUTH}"
)

print()


for nomor, image_path in enumerate(
    image_files,
    start=1
):

    filename = image_path.stem


    print("-" * 70)

    print(
        f"CITRA {nomor:02d}: {image_path.name}"
    )

    print("-" * 70)


    # ======================================================
    # ORIGINAL
    # ======================================================

    original = cv2.imread(
        str(image_path)
    )


    if original is None:

        print(
            "Original      : GAGAL DIBACA"
        )

        continue


    original_result = lakukan_ocr(
        original
    )


    # ======================================================
    # MEAN FILTER
    # ======================================================

    mean_path = (
        MEAN_FOLDER
        /
        f"{filename}.png"
    )


    mean_result = ""


    if mean_path.exists():

        mean_image = cv2.imread(
            str(mean_path),
            cv2.IMREAD_GRAYSCALE
        )


        if mean_image is not None:

            mean_result = lakukan_ocr(
                mean_image
            )


    # ======================================================
    # MEDIAN FILTER
    # ======================================================

    median_path = (
        MEDIAN_FOLDER
        /
        f"{filename}.png"
    )


    median_result = ""


    if median_path.exists():

        median_image = cv2.imread(
            str(median_path),
            cv2.IMREAD_GRAYSCALE
        )


        if median_image is not None:

            median_result = lakukan_ocr(
                median_image
            )


    # ======================================================
    # GAUSSIAN FILTER
    # ======================================================

    gaussian_path = (
        GAUSSIAN_FOLDER
        /
        f"{filename}.png"
    )


    gaussian_result = ""


    if gaussian_path.exists():

        gaussian_image = cv2.imread(
            str(gaussian_path),
            cv2.IMREAD_GRAYSCALE
        )


        if gaussian_image is not None:

            gaussian_result = lakukan_ocr(
                gaussian_image
            )


    # ======================================================
    # SHARPENING
    # ======================================================

    sharpening_path = (
        SHARPENING_FOLDER
        /
        f"{filename}.png"
    )


    sharpening_result = ""


    if sharpening_path.exists():

        sharpening_image = cv2.imread(
            str(sharpening_path),
            cv2.IMREAD_GRAYSCALE
        )


        if sharpening_image is not None:

            sharpening_result = lakukan_ocr(
                sharpening_image
            )


    # ======================================================
    # DATA HASIL
    # ======================================================

    hasil_ocr.append({

        "nomor": nomor,

        "filename": image_path.name,

        "original": original_result,

        "mean": mean_result,

        "median": median_result,

        "gaussian": gaussian_result,

        "sharpening": sharpening_result

    })


    # ======================================================
    # TAMPILKAN DI TERMINAL
    # ======================================================

    print(
        f"Ground Truth : "
        f"{GROUND_TRUTH}"
    )

    print()

    print(
        f"Original      : "
        f"{original_result if original_result else 'Tidak terbaca'}"
    )

    print(
        f"Mean Filter   : "
        f"{mean_result if mean_result else 'Tidak terbaca'}"
    )

    print(
        f"Median Filter : "
        f"{median_result if median_result else 'Tidak terbaca'}"
    )

    print(
        f"Gaussian      : "
        f"{gaussian_result if gaussian_result else 'Tidak terbaca'}"
    )

    print(
        f"Sharpening    : "
        f"{sharpening_result if sharpening_result else 'Tidak terbaca'}"
    )

    print()


# ==========================================================
# BUAT FOLDER HASIL
# ==========================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SIMPAN HASIL KE TXT
# ==========================================================

with open(
    OCR_RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        "HASIL OCR NOMOR IJAZAH\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"GROUND TRUTH:\n"
        f"{GROUND_TRUTH}\n\n"
    )


    for data in hasil_ocr:

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"CITRA {data['nomor']:02d}\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"File : {data['filename']}\n\n"
        )

        file.write(
            f"Ground Truth  : "
            f"{GROUND_TRUTH}\n"
        )

        file.write(
            f"Original      : "
            f"{data['original'] if data['original'] else 'Tidak terbaca'}\n"
        )

        file.write(
            f"Mean Filter   : "
            f"{data['mean'] if data['mean'] else 'Tidak terbaca'}\n"
        )

        file.write(
            f"Median Filter : "
            f"{data['median'] if data['median'] else 'Tidak terbaca'}\n"
        )

        file.write(
            f"Gaussian      : "
            f"{data['gaussian'] if data['gaussian'] else 'Tidak terbaca'}\n"
        )

        file.write(
            f"Sharpening    : "
            f"{data['sharpening'] if data['sharpening'] else 'Tidak terbaca'}\n"
        )

        file.write("\n")


    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        "OCR SELESAI\n"
    )

    file.write(
        "=" * 70 + "\n"
    )


# ==========================================================
# SELESAI
# ==========================================================

print("=" * 70)

print(
    "OCR SELESAI"
)

print("=" * 70)

print()

print(
    "Ground Truth:"
)

print(
    GROUND_TRUTH
)

print()

print(
    "Hasil OCR disimpan di:"
)

print(
    OCR_RESULT_FILE
)

print()

print(
    "Buka file tersebut untuk melihat hasil OCR."
)