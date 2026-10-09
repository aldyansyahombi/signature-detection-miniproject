from pathlib import Path
import re
from difflib import SequenceMatcher

# ============================================================
# KONFIGURASI
# ============================================================

FILE_OCR = Path("hasil/hasil_ocr.txt")
FILE_HASIL = Path("hasil/evaluasi_ocr.txt")

# Ground Truth lengkap
GROUND_TRUTH = "Nomor ijazah: 571012022000056"


# ============================================================
# FUNGSI NORMALISASI TEKS
# ============================================================

def normalisasi_teks(teks):
    """
    Menghapus semua spasi dari teks.

    Spasi TIDAK dihitung sebagai karakter
    dalam evaluasi OCR.
    """

    if teks is None:
        return ""

    # Hapus semua jenis spasi
    teks = re.sub(r"\s+", "", teks)

    return teks


# ============================================================
# FUNGSI HITUNG KARAKTER BENAR
# ============================================================

def hitung_akurasi(hasil_ocr, ground_truth):

    if hasil_ocr == "Tidak terbaca" or hasil_ocr == "":
        return 0, 0.0

    # Hapus spasi dari kedua teks
    hasil_ocr = normalisasi_teks(hasil_ocr)
    ground_truth = normalisasi_teks(ground_truth)

    # Jika setelah menghapus spasi tidak ada teks
    if hasil_ocr == "":
        return 0, 0.0

    # ========================================================
    # MENCARI KARAKTER YANG SAMA
    # ========================================================

    matcher = SequenceMatcher(
        None,
        ground_truth,
        hasil_ocr
    )

    matching_blocks = matcher.get_matching_blocks()

    karakter_benar = 0

    for block in matching_blocks:
        karakter_benar += block.size

    # ========================================================
    # HITUNG AKURASI
    # ========================================================

    total_karakter = len(ground_truth)

    akurasi = (
        karakter_benar / total_karakter
    ) * 100

    # Tidak boleh lebih dari 100%
    if akurasi > 100:
        akurasi = 100.0

    return karakter_benar, akurasi


# ============================================================
# CEK FILE OCR
# ============================================================

if not FILE_OCR.exists():

    print(f"File tidak ditemukan: {FILE_OCR}")
    print("Jalankan terlebih dahulu:")
    print("python ocr.py")

    exit()


# ============================================================
# BACA HASIL OCR
# ============================================================

text = FILE_OCR.read_text(
    encoding="utf-8"
)


# ============================================================
# PARSING HASIL OCR
# ============================================================

data = []

# Mencari semua CITRA 01 sampai CITRA 09
matches_citra = list(
    re.finditer(
        r"CITRA\s+(\d+)",
        text
    )
)


for index, match_citra in enumerate(matches_citra):

    nomor_citra = int(
        match_citra.group(1)
    )

    # --------------------------------------------------------
    # Menentukan batas blok citra
    # --------------------------------------------------------

    start = match_citra.start()

    if index + 1 < len(matches_citra):

        end = matches_citra[
            index + 1
        ].start()

    else:

        end = len(text)

    blok = text[start:end]


    # ========================================================
    # METODE OCR
    # ========================================================

    metode_dict = {

        "Original":
            r"^\s*Original\s*:\s*(.+)$",

        "Mean Filter":
            r"^\s*Mean Filter\s*:\s*(.+)$",

        "Median Filter":
            r"^\s*Median Filter\s*:\s*(.+)$",

        "Gaussian":
            r"^\s*Gaussian\s*:\s*(.+)$",

        "Sharpening":
            r"^\s*Sharpening\s*:\s*(.+)$"
    }


    for metode, pola in metode_dict.items():

        match = re.search(
            pola,
            blok,
            re.MULTILINE
        )


        if match:

            hasil_ocr = match.group(1).strip()

        else:

            hasil_ocr = "Tidak terbaca"


        # Jika kosong
        if hasil_ocr == "":

            hasil_ocr = "Tidak terbaca"


        # ====================================================
        # HITUNG KARAKTER BENAR DAN AKURASI
        # ====================================================

        karakter_benar, akurasi = hitung_akurasi(
            hasil_ocr,
            GROUND_TRUTH
        )


        # ====================================================
        # SIMPAN DATA
        # ====================================================

        data.append({

            "citra":
                nomor_citra,

            "metode":
                metode,

            "hasil_ocr":
                hasil_ocr,

            "karakter_benar":
                karakter_benar,

            "akurasi":
                akurasi
        })


# ============================================================
# KONFIGURASI TABEL
# ============================================================

metode_list = [

    "Original",
    "Mean Filter",
    "Median Filter",
    "Gaussian",
    "Sharpening"
]


# Lebar kolom
LEBAR_METODE = 18
LEBAR_OCR = 45
LEBAR_BENAR = 18
LEBAR_AKURASI = 12


GARIS_TABEL = (
    "+"
    + "-" * LEBAR_METODE
    + "+"
    + "-" * LEBAR_OCR
    + "+"
    + "-" * LEBAR_BENAR
    + "+"
    + "-" * LEBAR_AKURASI
    + "+"
)


# ============================================================
# FUNGSI BARIS TABEL
# ============================================================

def baris_tabel(
    metode,
    hasil_ocr,
    karakter_benar,
    akurasi
):

    # Jika hasil OCR terlalu panjang,
    # potong agar tabel tetap rapi
    if len(hasil_ocr) > LEBAR_OCR - 2:

        hasil_ocr = (
            hasil_ocr[
                :LEBAR_OCR - 5
            ]
            + "..."
        )


    return (
        "| "
        + metode.ljust(
            LEBAR_METODE - 2
        )
        + "| "
        + hasil_ocr.ljust(
            LEBAR_OCR - 2
        )
        + "| "
        + str(
            karakter_benar
        ).ljust(
            LEBAR_BENAR - 2
        )
        + "| "
        + f"{akurasi:.2f}%".ljust(
            LEBAR_AKURASI - 2
        )
        + "|"
    )


def header_tabel():

    return (
        "| "
        + "Metode".ljust(
            LEBAR_METODE - 2
        )
        + "| "
        + "Hasil OCR".ljust(
            LEBAR_OCR - 2
        )
        + "| "
        + "Karakter Benar".ljust(
            LEBAR_BENAR - 2
        )
        + "| "
        + "Akurasi".ljust(
            LEBAR_AKURASI - 2
        )
        + "|"
    )


# ============================================================
# BUAT HASIL EVALUASI
# ============================================================

hasil = []

GARIS = "=" * 100


# ============================================================
# HASIL SETIAP CITRA
# ============================================================

for nomor_citra in range(1, 10):

    hasil.append(GARIS)

    hasil.append(
        "HASIL EVALUASI OCR"
    )

    hasil.append(GARIS)

    hasil.append("")

    hasil.append(
        f"Ground Truth : {GROUND_TRUTH}"
    )

    hasil.append("")

    hasil.append(
        "Spasi tidak dihitung sebagai karakter."
    )

    hasil.append(
        "Akurasi dihitung berdasarkan karakter "
        "yang berhasil dikenali sesuai dengan "
        "Ground Truth."
    )

    hasil.append("")

    hasil.append(
        f"CITRA {nomor_citra:02d}"
    )

    hasil.append("")

    # --------------------------------------------------------
    # HEADER TABEL
    # --------------------------------------------------------

    hasil.append(
        GARIS_TABEL
    )

    hasil.append(
        header_tabel()
    )

    hasil.append(
        GARIS_TABEL
    )


    # --------------------------------------------------------
    # DATA CITRA
    # --------------------------------------------------------

    data_citra = [

        item

        for item in data

        if item["citra"] == nomor_citra
    ]


    # --------------------------------------------------------
    # TAMPILKAN SETIAP METODE
    # --------------------------------------------------------

    for metode in metode_list:

        item_ditemukan = None


        for item in data_citra:

            if item["metode"] == metode:

                item_ditemukan = item

                break


        # Jika data tidak ditemukan
        if item_ditemukan is None:

            hasil_ocr = "Tidak terbaca"

            karakter_benar = 0

            akurasi = 0.0

        else:

            hasil_ocr = (
                item_ditemukan[
                    "hasil_ocr"
                ]
            )

            karakter_benar = (
                item_ditemukan[
                    "karakter_benar"
                ]
            )

            akurasi = (
                item_ditemukan[
                    "akurasi"
                ]
            )


        hasil.append(
            baris_tabel(
                metode,
                hasil_ocr,
                karakter_benar,
                akurasi
            )
        )


    hasil.append(
        GARIS_TABEL
    )

    hasil.append("")
    hasil.append("")


# ============================================================
# REKAP AKURASI BERDASARKAN METODE
# ============================================================

hasil.append(GARIS)

hasil.append(
    "REKAP AKURASI BERDASARKAN METODE"
)

hasil.append(GARIS)

hasil.append("")


# ============================================================
# KONFIGURASI TABEL REKAP
# ============================================================

R_METODE = 20
R_BENAR = 24
R_TOTAL = 18
R_AKURASI = 12


GARIS_REKAP = (
    "+"
    + "-" * R_METODE
    + "+"
    + "-" * R_BENAR
    + "+"
    + "-" * R_TOTAL
    + "+"
    + "-" * R_AKURASI
    + "+"
)


hasil.append(
    GARIS_REKAP
)


hasil.append(
    "| "
    + "Metode".ljust(
        R_METODE - 2
    )
    + "| "
    + "Total Karakter Benar".ljust(
        R_BENAR - 2
    )
    + "| "
    + "Total Karakter".ljust(
        R_TOTAL - 2
    )
    + "| "
    + "Akurasi".ljust(
        R_AKURASI - 2
    )
    + "|"
)


hasil.append(
    GARIS_REKAP
)


# ============================================================
# TOTAL KARAKTER
# ============================================================

# Ground Truth setelah spasi dihapus:
# NomorIjazah:571012022000056
#
# Jumlah = 27 karakter
total_karakter_per_citra = len(
    normalisasi_teks(GROUND_TRUTH)
)

total_karakter = (
    9 * total_karakter_per_citra
)


# ============================================================
# REKAP SETIAP METODE
# ============================================================

for metode in metode_list:

    data_metode = [

        item

        for item in data

        if item["metode"] == metode
    ]


    total_benar = sum(

        item["karakter_benar"]

        for item in data_metode
    )


    if total_karakter > 0:

        akurasi_total = (
            total_benar
            / total_karakter
        ) * 100

    else:

        akurasi_total = 0.0


    hasil.append(

        "| "
        + metode.ljust(
            R_METODE - 2
        )
        + "| "
        + str(
            total_benar
        ).ljust(
            R_BENAR - 2
        )
        + "| "
        + str(
            total_karakter
        ).ljust(
            R_TOTAL - 2
        )
        + "| "
        + f"{akurasi_total:.2f}%".ljust(
            R_AKURASI - 2
        )
        + "|"
    )


hasil.append(
    GARIS_REKAP
)


# ============================================================
# SIMPAN HASIL
# ============================================================

FILE_HASIL.parent.mkdir(
    parents=True,
    exist_ok=True
)


FILE_HASIL.write_text(
    "\n".join(hasil),
    encoding="utf-8"
)


# ============================================================
# TAMPILKAN KE TERMINAL
# ============================================================

print(
    "\n".join(hasil)
)


print(
    "\n" + GARIS
)

print(
    "Evaluasi OCR selesai."
)

print(
    f"Hasil disimpan di: {FILE_HASIL}"
)

print(
    GARIS
)