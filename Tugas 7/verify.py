#!/usr/bin/env python3
"""CLI: python verify.py ijazah_001.jpg [--debug out_dir] [--enhancement NAMA] [--verbose]"""
import argparse
import json
from ijazah import pipeline
from ijazah.enhance import METHODS


def main():
    ap = argparse.ArgumentParser(description="Verifikasi ijazah: OCR nomor + deteksi tanda tangan")
    ap.add_argument("image")
    ap.add_argument("--enhancement", choices=list(METHODS), help="metode enhancement ROI nomor")
    ap.add_argument("--debug", metavar="DIR", help="simpan citra tiap tahap pipeline")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    r = pipeline.run(a.image, a.enhancement, a.debug)
    print(f"Nomor Ijazah : {r['nomor_ijazah']}")
    print(f"Tanda Tangan : {r['tanda_tangan']}")
    if a.verbose:
        print(json.dumps({k: r[k] for k in ("ocr_raw", "signature_info", "rotation", "enhancement")},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
