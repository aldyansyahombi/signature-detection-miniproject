import cv2

from . import config, enhance, ocr, preprocess, signature


def run(path, enhancement=None, debug_dir=None):
    enhancement = enhancement or config.DEFAULT_OCR_ENHANCEMENT
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)

    upright, rot = preprocess.auto_orient(bgr)            # koreksi orientasi
    upright = preprocess.standardize(upright)
    gray = preprocess.to_gray(upright)                    # Grayscale
    enh = preprocess.global_enhance(gray)                 # Image Enhancement (global)

    # --- cabang 1: nomor ijazah ---
    roi_num = ocr.get_number_roi(enh)                     # Area Nomor
    roi_num_e = enhance.METHODS[enhancement](roi_num)     # Enhancement (ROI)
    number, raw = ocr.read_number(roi_num_e)              # OCR

    # --- cabang 2: tanda tangan (Rektor) ---
    roi_sig = signature.get_signature_roi(enh, "rektor")  # Area Tanda Tangan
    present, info, dbg = signature.detect_signature(roi_sig, return_debug=True)  # Threshold+Morfologi+Deteksi

    if debug_dir:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/1_gray.jpg", gray)
        cv2.imwrite(f"{debug_dir}/2_enhanced.jpg", enh)
        cv2.imwrite(f"{debug_dir}/3_roi_nomor.png", roi_num)
        cv2.imwrite(f"{debug_dir}/4_roi_nomor_enhanced.png", roi_num_e)
        cv2.imwrite(f"{debug_dir}/5_roi_ttd.png", roi_sig)
        for k, v in dbg.items():
            cv2.imwrite(f"{debug_dir}/6_ttd_{k}.png", v)

    return {"nomor_ijazah": number, "ocr_raw": raw, "tanda_tangan": "PRESENT" if present else "ABSENT",
            "signature_info": info, "rotation": rot, "enhancement": enhancement}
