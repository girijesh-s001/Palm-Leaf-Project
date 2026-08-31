"""
char_seg.py
-----------
Character segmentation for Tamil palm-leaf manuscript images.

Public API:
    get_initial_blocks(line_binary)          -> list of block dicts
    adaptive_split_and_save(...)             -> (char_count, metrics_dict)
"""

import os
import cv2
import numpy as np


def get_initial_blocks(line_binary):
    """
    Scans the vertical ink projection of a binary line image and returns
    initial character block bounding boxes (x, y, w, h).
    """
    _, w_line = line_binary.shape
    v_proj = np.sum(line_binary == 255, axis=0)
    in_char = False
    start_x = 0
    blocks = []

    for x, has_ink in enumerate(v_proj > 0):
        if has_ink and not in_char:
            start_x, in_char = x, True
        elif not has_ink and in_char:
            in_char = False
            blk = _slice_to_block(line_binary, start_x, x)
            if blk:
                blocks.append(blk)

    if in_char:
        blk = _slice_to_block(line_binary, start_x, w_line)
        if blk:
            blocks.append(blk)

    return blocks


def _slice_to_block(binary, x0, x1):
    rows, _ = np.where(binary[:, x0:x1] == 255)
    if len(rows) == 0:
        return None
    return {"x": x0, "y": int(rows.min()), "w": x1 - x0, "h": int(rows.max() - rows.min() + 1)}


def _boxes_intersect(a, b):
    return not (
        b["x"] >= a["x"] + a["w"]
        or a["x"] >= b["x"] + b["w"]
        or b["y"] >= a["y"] + a["h"]
        or a["y"] >= b["y"] + b["h"]
    )


def _union_box(a, b, source):
    x = min(a["x"], b["x"])
    y = min(a["y"], b["y"])
    w = max(a["x"] + a["w"], b["x"] + b["w"]) - x
    h = max(a["y"] + a["h"], b["y"] + b["h"]) - y
    return {"x": x, "y": y, "w": w, "h": h, "crop": source[y:y + h, x:x + w], "is_merged": False}


def merge_overlapping_boxes(chars, line_source):
    """Merges intersecting bounding boxes iteratively."""
    if not chars:
        return []

    chars = sorted(chars, key=lambda c: c["x"])
    changed = True

    while changed:
        changed = False
        merged, used = [], set()
        for i, a in enumerate(chars):
            if i in used:
                continue
            for j in range(i + 1, len(chars)):
                if j not in used and _boxes_intersect(a, chars[j]):
                    a = _union_box(a, chars[j], line_source)
                    used.add(j)
                    changed = True
            merged.append(a)
            used.add(i)
        chars = merged

    return chars


def adaptive_split_and_save(line_binary, line_blocks, avg_w, avg_h, out_dir, line_source=None):
    """
    Refines detected blocks using contour analysis, merges overlapping boxes,
    and returns segmented characters and bounding metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    if line_source is None:
        line_source = line_binary

    raw_chars = []
    for block in line_blocks:
        bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]
        crop = line_binary[by:by + bh, bx:bx + bw]
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sub = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                gx, gy = bx + cx, by + cy
                sub.append({
                    "x": gx,
                    "y": gy,
                    "w": cw,
                    "h": ch,
                    "crop": line_source[gy:gy + ch, gx:gx + cw],
                    "is_merged": False,
                })
        raw_chars.extend(sorted(sub, key=lambda s: s["x"]))

    final_chars = merge_overlapping_boxes(raw_chars, line_source)

    merge_threshold = 1.2
    for c in final_chars:
        if c["w"] > merge_threshold * avg_w or c["h"] > merge_threshold * avg_h:
            c["is_merged"] = True

    final_chars.sort(key=lambda c: (c["x"], c["y"]))

    metrics = {k: [] for k in ("bboxes", "x_min", "x_max", "y_min", "y_max",
                                "widths", "heights", "areas", "avgs", "is_merged", "crops")}
    for c in final_chars:
        crop = c["crop"]
        metrics["bboxes"].append((c["x"], c["y"], c["w"], c["h"]))
        metrics["x_min"].append(c["x"])
        metrics["x_max"].append(c["x"] + c["w"])
        metrics["y_min"].append(c["y"])
        metrics["y_max"].append(c["y"] + c["h"])
        metrics["widths"].append(c["w"])
        metrics["heights"].append(c["h"])
        metrics["areas"].append(c["w"] * c["h"])
        metrics["avgs"].append(float(np.mean(crop)))
        metrics["is_merged"].append(c["is_merged"])
        metrics["crops"].append(crop)

    return len(final_chars), metrics
