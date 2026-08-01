"""
ocr_pipeline.py
================
Unified pipeline: LINE_SEG character segmentation -> CNN prediction.

Preserves the spatial (x, y) reading order of characters exactly as they
appear in the input image (lines top-to-bottom, characters left-to-right
within each line).

Usage:
    from ocr_pipeline import run_ocr
    result = run_ocr("path/to/image.png")
    print(result["text"])
"""

import os
import sys
import cv2
import numpy as np

# ---------- Path setup ----------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LINE_SEG = os.path.join(_THIS_DIR, "LINE_SEG")
_YAML_ANN = os.path.join(_THIS_DIR, "yaml annotation")

for _p in [_LINE_SEG, _YAML_ANN]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------- Imports ----------
import char_seg
from line_seg import preprocess, estimate_height, detect_separators, dp_trace
from preprocess import preprocess_pipeline
from predict import load_trained_model

# ---------- Model singleton ----------
_MODEL = None
_CLASSES = None


def _get_model():
    global _MODEL, _CLASSES
    if _MODEL is None:
        _CLASSES = np.load(os.path.join(_YAML_ANN, "classes.npy"), allow_pickle=True)
        _MODEL = load_trained_model(
            classes_count=len(_CLASSES),
            model_path=os.path.join(_YAML_ANN, "cnn_model.h5")
        )
    return _MODEL, _CLASSES


def _predict_crop(crop_gray, model, classes):
    """Predict a single character from a grayscale crop."""
    bgr = (cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
           if len(crop_gray.shape) == 2 else crop_gray.copy())
    processed = preprocess_pipeline(bgr, target_size=(64, 64), invert=True)
    tensor = np.expand_dims(processed, axis=0)
    preds = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(preds))
    return str(classes[idx]), float(preds[idx]) * 100.0


def run_ocr(image_source):
    """
    Run end-to-end OCR on a palm-leaf document image.

    Parameters
    ----------
    image_source : str | np.ndarray
        File path or BGR numpy array.

    Returns
    -------
    dict: text, lines, char_list, line_vis, char_vis, image, error
    """
    model, classes = _get_model()

    # 1. Load
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is None:
            return {"error": "Could not read image: " + image_source, "text": ""}
    else:
        img = image_source.copy()

    h_img, w_img = img.shape[:2]

    # 2. Preprocess
    cleaned, thresh_inv, sharpened = preprocess(img, show=False)

    # 3. Line detection
    text_h = estimate_height(cleaned)
    reinforced = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(text_h * 2, 30), 1))
    )
    seps = detect_separators(cleaned, text_h)
    paths = [dp_trace(reinforced, y) for y in seps]

    if len(paths) > 1:
        paths.sort(key=lambda p: np.median(p))
        merged = [paths[0]]
        for p in paths[1:]:
            if np.mean(np.abs(p.astype(float) - merged[-1].astype(float))) > max(text_h // 3, 6):
                merged.append(p)
        paths = merged

    # 4. Build line strips
    h, w = cleaned.shape
    inner = sorted(paths, key=lambda p: np.median(p))
    full_paths = (
        [np.zeros(w, dtype=np.int32)]
        + inner
        + [np.full(w, h - 1, dtype=np.int32)]
    )

    line_data, all_initial_blocks = [], []

    for i in range(len(full_paths) - 1):
        if np.median(full_paths[i + 1]) - np.median(full_paths[i]) < 8:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.vstack((
            np.column_stack((np.arange(w), full_paths[i])),
            np.flipud(np.column_stack((np.arange(w), full_paths[i + 1])))
        )).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        y_idx, x_idx = np.where(mask == 255)
        if len(y_idx) == 0 or np.sum(cleaned[mask == 255] == 255) < 50:
            continue

        lb = np.zeros((h, w), dtype=np.uint8)
        lb[mask == 255] = cleaned[mask == 255]
        lt = np.zeros((h, w), dtype=np.uint8)
        lt[mask == 255] = thresh_inv[mask == 255]

        y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
        x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
        lcb = lb[y_min:y_max + 1, x_min:x_max + 1]
        lct = lt[y_min:y_max + 1, x_min:x_max + 1]

        ib = char_seg.get_initial_blocks(lcb)
        all_initial_blocks.extend(ib)
        line_data.append({
            "bin": lcb, "thresh": lct,
            "y_min": y_min, "x_min": x_min, "initial_blocks": ib
        })

    if not all_initial_blocks:
        return {
            "error": "No text detected.", "text": "", "lines": [], "char_list": [],
            "line_vis": img.copy(), "char_vis": img.copy(), "image": img
        }

    # 5. Global stats
    avg_w = float(np.median([b["w"] for b in all_initial_blocks]))
    avg_h = float(np.median([b["h"] for b in all_initial_blocks]))

    # 6. Per-line segmentation
    tmp_dir = os.path.join(_LINE_SEG, "_tmp_chars")
    all_chars = []

    for li, L in enumerate(line_data):
        if L["initial_blocks"]:
            law = float(np.median([b["w"] for b in L["initial_blocks"]]))
            lah = float(np.median([b["h"] for b in L["initial_blocks"]]))
            if len(L["initial_blocks"]) < 5:
                law = (law + avg_w) / 2
                lah = (lah + avg_h) / 2
        else:
            law, lah = avg_w, avg_h

        cnt, metrics = char_seg.adaptive_split_and_save(
            L["bin"], L["initial_blocks"], law, lah,
            os.path.join(tmp_dir, "line_" + str(li)),
            line_source=L["thresh"]
        )
        for j in range(cnt):
            all_chars.append({
                "line_idx": li,
                "abs_x": L["x_min"] + metrics["x_min"][j],
                "abs_y": L["y_min"] + metrics["y_min"][j],
                "w": metrics["widths"][j],
                "h": metrics["heights"][j],
                "area": metrics["areas"][j],
                "crop": metrics["crops"][j],
                "is_merged": metrics["is_merged"][j],
            })

    # 7. Two-stage split
    if all_chars:
        ag_w = sum(c["w"] for c in all_chars) / len(all_chars)
        ag_h = sum(c["h"] for c in all_chars) / len(all_chars)
        s1w, s1h = 1.6 * ag_w, 1.6 * ag_h
        sl = [c for c in all_chars if c["w"] > s1w or c["h"] > s1h]
        ftw = 1.08 * sum(c["w"] for c in sl) / len(sl) if sl else 9999
        cf = []
        for c in all_chars:
            sp = (c["w"] > s1w or c["h"] > s1h) and c["w"] > ftw
            if sp:
                crop = c["crop"]
                hc, wc = crop.shape
                vh = np.sum(crop == 255, axis=0)
                sx, ex = int(wc * 0.35), int(wc * 0.65)
                if ex > sx:
                    sa = vh[sx:ex]
                    mv = int(np.min(sa))
                    si = sx + int(np.where(sa == mv)[0][0])
                else:
                    mv, si = 0, wc // 2
                pl = int(np.max(vh[:sx])) if sx > 0 else 0
                pr = int(np.max(vh[ex:])) if ex < wc else 0
                mnp = min(pl, pr) if pl > 0 and pr > 0 else 0
                ar = wc / hc if hc > 0 else 0
                ij = ar > 1.6 and mnp > 0.5 * hc and mv < 0.3 * mnp
                if ij:
                    cf.append({**c, "w": si, "crop": crop[:, :si],
                               "abs_x": c["abs_x"], "is_split": True})
                    cf.append({**c, "w": wc - si, "crop": crop[:, si:],
                               "abs_x": c["abs_x"] + si, "is_split": True})
                    continue
            c["is_split"] = False
            cf.append(c)
    else:
        cf = []

    # 8. Sort by (line_idx, abs_x) -- preserves reading order
    cf.sort(key=lambda c: (c["line_idx"], c["abs_x"]))

    # 9. CNN prediction
    char_list = []
    lines_text = {}
    for c in cf:
        label, conf = _predict_crop(255 - c["crop"], model, classes)
        entry = {
            "line_idx": c["line_idx"],
            "abs_x": c["abs_x"], "abs_y": c["abs_y"],
            "w": c["w"], "h": c["h"],
            "label": label, "confidence": round(conf, 2),
            "is_merged": c.get("is_merged", False),
            "is_split": c.get("is_split", False),
        }
        char_list.append(entry)
        lines_text.setdefault(c["line_idx"], []).append(label)

    # 10. Assemble text
    text_lines = ["".join(lines_text[li]) for li in sorted(lines_text)]
    final_text = "\n".join(text_lines)

    lines_summary = [
        {"line": li + 1,
         "char_count": len([e for e in char_list if e["line_idx"] == li]),
         "text": "".join(lines_text[li])}
        for li in sorted(lines_text)
    ]

    # 10b. Accuracy metrics (same formula as line_seg.py report)
    total_final  = len(cf)
    split_chars  = sum(1 for c in cf if c.get("is_split"))
    merged_chars = sum(1 for c in cf if c.get("is_merged") and not c.get("is_split"))
    good_chars   = total_final - merged_chars - split_chars

    # aspect-ratio quality: 0.3 – 2.0 is considered OK
    ar_ok  = sum(1 for c in cf if 0.3 <= (c["w"] / c["h"] if c["h"] > 0 else 0) <= 2.0)
    ar_bad = total_final - ar_ok

    # width within 80 % of global average
    if cf:
        aw = sum(c["w"] for c in cf) / len(cf)
        w_ok  = sum(1 for c in cf if 0.2 * aw <= c["w"] <= 1.8 * aw)
    else:
        w_ok = 0
    w_bad = total_final - w_ok

    # Score = (good + split) / total  (splits were resolved, so they count as correct)
    accuracy_score = round((good_chars + split_chars) / total_final * 100, 2) if total_final > 0 else 0.0

    accuracy = {
        "score":                accuracy_score,
        "cnn_dataset_accuracy": 89.19,
        "cnn_test_accuracy":    48.30,
        "total_chars":          total_final,
        "good_chars":           good_chars,
        "split_chars":          split_chars,
        "merged_chars":         merged_chars,
        "ar_ok":                ar_ok,
        "ar_bad":               ar_bad,
        "width_ok":             w_ok,
        "width_bad":            w_bad,
    }
    line_vis = img.copy()
    for p in inner:
        cv2.polylines(
            line_vis,
            [np.column_stack((np.arange(len(p)), p)).astype(np.int32)],
            False, (0, 0, 255), 2
        )

    char_vis = np.full((h_img, w_img, 3), 255, dtype=np.uint8)
    for e in char_list:
        x, y, bw, bh = e["abs_x"], e["abs_y"], e["w"], e["h"]
        col = (0, 0, 255) if e["is_merged"] or e["is_split"] else (255, 0, 0)
        cv2.rectangle(char_vis, (x, y), (x + bw, y + bh), col, 1)
        cv2.putText(char_vis, e["label"], (x, max(0, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "error": None, "text": final_text,
        "lines": lines_summary, "char_list": char_list,
        "accuracy": accuracy,
        "line_vis": line_vis, "char_vis": char_vis, "image": img,
    }


# ---------- CLI ----------
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Select Palm Leaf Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if not img_path:
        print("No image selected. Exiting.")
        sys.exit(0)

    print("Processing:", img_path)
    result = run_ocr(img_path)

    if result.get("error"):
        print("ERROR:", result["error"])
    else:
        print("\n" + "=" * 60)
        print("  RECOGNISED TEXT (reading order preserved)")
        print("=" * 60)
        print(result["text"])
        print("=" * 60)
        print("Lines:", len(result["lines"]), " Chars:", len(result["char_list"]))
        for ln in result["lines"]:
            print("  Line", ln["line"], ":", ln["char_count"], "chars ->", ln["text"])
        acc = result["accuracy"]
        print("\n" + "=" * 60)
        print(f"  SEGMENTATION ACCURACY: {acc['score']:.2f}%")
        print(f"  Total: {acc['total_chars']}  Good: {acc['good_chars']}  "
              f"Split(resolved): {acc['split_chars']}  Merged(flagged): {acc['merged_chars']}")
        print(f"  Aspect-ratio OK: {acc['ar_ok']}  Bad: {acc['ar_bad']}")
        print(f"  Width within avg: {acc['width_ok']}  Out: {acc['width_bad']}")
        print("=" * 60)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out = os.path.join(os.path.dirname(img_path), "output", base)
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, "line_segmented.png"), result["line_vis"])
        cv2.imwrite(os.path.join(out, "char_segmented.png"), result["char_vis"])
        txt_path = os.path.join(out, "recognized_text.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print("\nOutputs saved to:", out)
        print("Text saved to   :", txt_path)
