import os
import sys
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import char_seg

def preprocess(img, is_crop=False, show=False):
    """
    Converts a BGR image to a clean binary mask for segmentation.
    Returns (cleaned, thresh_inv, sharpened).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpened = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 11)
    inv = ~thresh

    min_area = 2 if is_crop else 10
    max_area = 100000 if is_crop else 14000
    cc_thresh = 10 if is_crop else 800

    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = thresh.copy()
    for cnt in contours:
        if not (min_area < cv2.contourArea(cnt) < max_area):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(cleaned, (x, y), (x + w, y + h), 255, -1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(~cleaned, connectivity=8)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if (area > cc_thresh and not is_crop) or (area <= cc_thresh and is_crop):
            cleaned[labels == i] = 255

    if show:
        cv2.imshow("Preprocess Input", img)
        cv2.imshow("Preprocess Cleaned", ~cleaned)

    return ~cleaned, ~thresh, sharpened


def estimate_height(binary):
    """Estimates typical character height from connected-component statistics."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights = [
        stats[i, cv2.CC_STAT_HEIGHT]
        for i in range(1, len(stats))
        if 5 < stats[i, cv2.CC_STAT_HEIGHT] < 80 and stats[i, cv2.CC_STAT_WIDTH] > 3
    ]
    return max(int(np.median(heights)), 10) if heights else 25


def detect_separators(binary, text_h):
    """Detects line separator valley rows from the vertical ink projection."""
    profile = np.sum(binary == 255, axis=1).astype(np.float64)
    smooth = gaussian_filter1d(profile, sigma=max(text_h / 6, 1.5))
    min_dist = max(int(text_h * 0.6), 8)
    best_v = []

    for prominence_frac in [0.15, 0.1, 0.05, 0.02, 0.01]:
        peaks, _ = find_peaks(smooth, distance=min_dist, prominence=prominence_frac * np.max(smooth))
        if len(peaks) < 2:
            continue
        valleys = []
        for j in range(len(peaks) - 1):
            seg = smooth[peaks[j]:peaks[j + 1] + 1]
            v = peaks[j] + int(np.argmin(seg))
            left_h, right_h = smooth[peaks[j]], smooth[peaks[j + 1]]
            if (min(left_h, right_h) - smooth[v]) > 0.02 * min(left_h, right_h):
                valleys.append(v)
        if len(valleys) > len(best_v):
            best_v = valleys

    bg = (binary == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(bg, cv2.DIST_L2, 5)
    row_dist = np.mean(dist, axis=1)
    r = max(text_h // 3, 5)
    return [
        y - r + int(np.argmax(row_dist[max(0, y - r):min(len(row_dist), y + r + 1)]))
        for y in best_v
    ]


def dp_trace(binary, y_start):
    """Traces an optimal seam path across the image avoiding ink pixels."""
    h, w = binary.shape
    bg = (binary == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(bg, cv2.DIST_L2, 3)

    cost = (np.max(dist) + 1 - dist).astype(np.float32)
    cost[binary > 0] += 250

    path = np.zeros(w, dtype=np.int32)
    path[0] = int(np.clip(y_start, 0, h - 1))

    for x in range(1, w):
        prev_y = path[x - 1]
        best_y = prev_y
        min_c = cost[prev_y, x]
        for dy in (-2, -1, 0, 1, 2):
            ny = prev_y + dy
            if 0 <= ny < h:
                c = cost[ny, x] + abs(dy) * 5 + abs(ny - y_start) * 0.1
                if c < min_c:
                    min_c, best_y = c, ny
        path[x] = best_y

    return path


def _merge_close_paths(paths, text_h):
    if len(paths) <= 1:
        return paths
    paths = sorted(paths, key=lambda p: np.median(p))
    merged = [paths[0]]
    for p in paths[1:]:
        gap = np.mean(np.abs(p.astype(float) - merged[-1].astype(float)))
        if gap > max(text_h // 3, 6):
            merged.append(p)
    return merged


def _build_line_strip(cleaned, thresh_inv, top_path, bot_path, min_ink=50):
    h, w = cleaned.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.vstack((
        np.column_stack((np.arange(w), top_path)),
        np.flipud(np.column_stack((np.arange(w), bot_path))),
    )).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)

    y_idx, x_idx = np.where(mask == 255)
    if len(y_idx) == 0 or np.sum(cleaned[mask == 255] == 255) < min_ink:
        return None

    y0, y1 = int(y_idx.min()), int(y_idx.max())
    x0, x1 = int(x_idx.min()), int(x_idx.max())

    lb = np.zeros((h, w), dtype=np.uint8)
    lb[mask == 255] = cleaned[mask == 255]
    lt = np.zeros((h, w), dtype=np.uint8)
    lt[mask == 255] = thresh_inv[mask == 255]

    return lb[y0:y1 + 1, x0:x1 + 1], lt[y0:y1 + 1, x0:x1 + 1], y0, x0


def process_image(path):
    """Segments a palm leaf manuscript image and writes visual results."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Cannot read image {path}")
        return

    print(f"Processing: {os.path.basename(path)}")
    cleaned, thresh_inv, _ = preprocess(img, show=False)

    text_h = estimate_height(cleaned)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(text_h * 2, 30), 1))
    reinforced = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    seps = detect_separators(cleaned, text_h)
    paths = [dp_trace(reinforced, y) for y in seps]
    paths = _merge_close_paths(paths, text_h)

    h, w = cleaned.shape
    inner = sorted(paths, key=lambda p: np.median(p))
    full_paths = [np.zeros(w, dtype=np.int32)] + inner + [np.full(w, h - 1, dtype=np.int32)]

    line_data, all_initial_blocks = [], []
    for i in range(len(full_paths) - 1):
        if np.median(full_paths[i + 1]) - np.median(full_paths[i]) < 8:
            continue
        strip = _build_line_strip(cleaned, thresh_inv, full_paths[i], full_paths[i + 1])
        if strip is None:
            continue
        lcb, lct, y_min, x_min = strip
        blocks = char_seg.get_initial_blocks(lcb)
        all_initial_blocks.extend(blocks)
        line_data.append({"bin": lcb, "thresh": lct, "y_min": y_min, "x_min": x_min, "initial_blocks": blocks})

    if not all_initial_blocks:
        print("No text detected in image.")
        return

    avg_w = float(np.median([b["w"] for b in all_initial_blocks]))
    avg_h = float(np.median([b["h"] for b in all_initial_blocks]))

    out = os.path.join("output", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, "cleaned.png"), 255 - cleaned)
    cv2.imwrite(os.path.join(out, "thresh.png"), 255 - thresh_inv)

    all_final_chars = []
    s_no = 0
    for i, L in enumerate(line_data):
        if L["initial_blocks"]:
            lw = float(np.median([b["w"] for b in L["initial_blocks"]]))
            lh = float(np.median([b["h"] for b in L["initial_blocks"]]))
            if len(L["initial_blocks"]) < 5:
                lw = (lw + avg_w) / 2
                lh = (lh + avg_h) / 2
        else:
            lw, lh = avg_w, avg_h

        line_num = i + 1
        cv2.imwrite(os.path.join(out, f"line_{line_num}.png"), 255 - L["thresh"])
        char_dir = os.path.join(out, f"line_{line_num}_chars")
        cnt, metrics = char_seg.adaptive_split_and_save(
            L["bin"], L["initial_blocks"], lw, lh, char_dir, line_source=L["thresh"]
        )
        for j in range(cnt):
            s_no += 1
            h_j = metrics["heights"][j]
            all_final_chars.append({
                "s_no": s_no,
                "line_idx": i,
                "local_bbox": metrics["bboxes"][j],
                "abs_x": L["x_min"] + metrics["x_min"][j],
                "abs_y": L["y_min"] + metrics["y_min"][j],
                "w": metrics["widths"][j],
                "h": h_j,
                "area": metrics["areas"][j],
                "aspect_ratio": round(metrics["widths"][j] / h_j, 2) if h_j > 0 else 0,
                "crop": metrics["crops"][j],
                "is_merged": metrics["is_merged"][j],
            })

    if not all_final_chars:
        print("No characters extracted.")
        return

    ag_w = sum(c["w"] for c in all_final_chars) / len(all_final_chars)
    ag_h = sum(c["h"] for c in all_final_chars) / len(all_final_chars)
    s1w, s1h = 1.6 * ag_w, 1.6 * ag_h

    shortlist = [c for c in all_final_chars if c["w"] > s1w or c["h"] > s1h]
    final_thresh_w = 1.08 * (sum(c["w"] for c in shortlist) / len(shortlist)) if shortlist else 9999

    final_char_dir = os.path.join(out, "final_characters")
    os.makedirs(final_char_dir, exist_ok=True)

    complete_list = []
    for c in all_final_chars:
        should_split = (c["w"] > s1w or c["h"] > s1h) and c["w"] > final_thresh_w
        if should_split:
            crop = c["crop"]
            h_c, w_c = crop.shape
            v_hist = np.sum(crop == 255, axis=0)
            sx, ex = int(w_c * 0.35), int(w_c * 0.65)

            if ex > sx:
                sa = v_hist[sx:ex]
                min_val = int(sa.min())
                si = sx + int(np.where(sa == min_val)[0][0])
            else:
                min_val, si = 0, w_c // 2

            pl = int(v_hist[:sx].max()) if sx > 0 else 0
            pr = int(v_hist[ex:].max()) if ex < w_c else 0
            mnp = min(pl, pr) if pl > 0 and pr > 0 else 0
            ar = w_c / h_c if h_c > 0 else 0
            is_joint = ar > 1.6 and mnp > 0.5 * h_c and min_val < 0.3 * mnp

            if is_joint:
                bx, by, _, bh = c["local_bbox"]
                crop_a = crop[:, :si]
                crop_b = crop[:, si:]
                complete_list.append({**c, "s_no": f"{c['s_no']}a", "w": si, "crop": crop_a,
                                      "abs_x": c["abs_x"], "is_split": True,
                                      "local_bbox": (bx, by, si, bh),
                                      "area": int((crop_a == 255).sum()),
                                      "aspect_ratio": round(si / h_c, 2) if h_c else 0})
                complete_list.append({**c, "s_no": f"{c['s_no']}b", "w": w_c - si, "crop": crop_b,
                                      "abs_x": c["abs_x"] + si, "is_split": True,
                                      "local_bbox": (bx + si, by, w_c - si, bh),
                                      "area": int((crop_b == 255).sum()),
                                      "aspect_ratio": round((w_c - si) / h_c, 2) if h_c else 0})
                continue
        c["is_split"] = False
        complete_list.append(c)

    for c in complete_list:
        cv2.imwrite(os.path.join(final_char_dir, f"char_{c['s_no']}.png"), c["crop"])

    total_f = len(complete_list)
    n_split = sum(1 for c in complete_list if c.get("is_split"))
    n_merged = sum(1 for c in complete_list if c.get("is_merged") and not c.get("is_split"))
    n_good = total_f - n_split - n_merged
    score = (n_good + n_split) / total_f * 100 if total_f else 0.0

    print(f"Extraction complete: {total_f} characters (Accuracy Score: {score:.2f}%)")

    h_img, w_img = img.shape[:2]
    line_vis = img.copy()
    char_vis = np.full((h_img, w_img, 3), 255, dtype=np.uint8)

    for p in inner:
        pts = np.column_stack((np.arange(len(p)), p)).astype(np.int32)
        cv2.polylines(line_vis, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    for c in complete_list:
        x, y, cw, ch = c["abs_x"], c["abs_y"], c["w"], c["h"]
        crop = c["crop"]
        ys, ye = max(0, y), min(h_img, y + ch)
        xs, xe = max(0, x), min(w_img, x + cw)
        roi = char_vis[ys:ye, xs:xe]
        roi_crop = crop[ys - y:ys - y + (ye - ys), xs - x:xs - x + (xe - xs)]
        if roi.shape[:2] == roi_crop.shape[:2]:
            roi[roi_crop == 255] = 0
        color = (0, 0, 255) if (c.get("is_split") or c.get("is_merged")) else (255, 0, 0)
        cv2.rectangle(char_vis, (x, y), (x + cw, y + ch), color, 1)
        cv2.putText(char_vis, str(c["s_no"]), (x, max(0, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    cv2.imwrite(os.path.join(out, "input_image.png"), img)
    cv2.imwrite(os.path.join(out, "line_segmented.png"), line_vis)
    cv2.imwrite(os.path.join(out, "char_segmented.png"), char_vis)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        img_path = filedialog.askopenfilename(
            title="Select Palm Leaf Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )

    if img_path:
        process_image(img_path)
    else:
        print("No image selected. Exiting.")