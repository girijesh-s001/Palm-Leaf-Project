"""
ocr_model_comparison.py
========================
Compare OCR models on the Tamil Palm Leaf annotated dataset.

Each model function follows the same 3-step structure:
  1. Import dependencies
  2. Insert dataset into the model
  3. Output accuracy compared to ground truth

Run:
  python ocr_model_comparison.py
  python ocr_model_comparison.py --images 10
"""

import os, re, sys, csv, time
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE_DIR = Path(__file__).parent
ANN_DIR  = BASE_DIR / "yaml annotation" / "dataset" / "annotations"
IMG_DIR  = BASE_DIR / "yaml annotation" / "dataset" / "images"
OUT_DIR  = BASE_DIR / "output" / "comparison"


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth():
    ground_truths = {}
    for yaml_file in sorted(ANN_DIR.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r"(labels:\s*)\n\[\]", r"\1 []", content)
        data = yaml.safe_load(content)
        chars = []
        for ann in data.get("annotations", []):
            bbox, labels = ann.get("bbox", []), ann.get("labels", [])
            if len(bbox) >= 4 and labels:
                label = str(labels[0]).strip()
                if label:
                    chars.append((int(bbox[0]), label))
        chars.sort(key=lambda c: c[0])
        ground_truths[yaml_file.stem] = "".join(lbl for _, lbl in chars)
    print(f"[Dataset] {len(ground_truths)} images | "
          f"{sum(len(v) for v in ground_truths.values())} ground-truth characters loaded")
    return ground_truths


def load_image(image_id):
    from PIL import Image as PILImage
    p = IMG_DIR / (image_id + ".jpg")
    if not p.exists():
        p = IMG_DIR / (image_id + ".png")
    bgr = cv2.imread(str(p))
    return bgr, PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_cer(predicted, ground_truth):
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    ref, hyp = ground_truth, predicted
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if ref[i-1] == hyp[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return min(dp[n] / len(ref), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Palm Leaf CNN (Project Baseline)
# ─────────────────────────────────────────────────────────────────────────────

def model_1_cnn(image_id, bgr, pil, ground_truth):

    # Step 1: Import dependencies
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from ocr_pipeline import run_ocr          # existing unmodified file

    # Step 2: Insert dataset image into the model
    result = run_ocr(bgr)
    predicted = result.get("text", "")

    # Step 3: Output accuracy compared to ground truth
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — PARSeq (Transformer Sequence Recognizer)
# ─────────────────────────────────────────────────────────────────────────────

def model_2_parseq(image_id, bgr, pil, ground_truth):

    # Step 1: Import dependencies
    import torch                              # pip install torch torchvision

    # Step 2: Insert dataset image into the model
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    model.eval()
    tensor = model.get_transform()(pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    pred, _ = model.tokenizer.decode(logits)
    predicted = pred[0] if pred else ""

    # Step 3: Output accuracy compared to ground truth
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — PP-OCRv5 (ta_PP-OCRv5_mobile_rec)
# ─────────────────────────────────────────────────────────────────────────────

def model_3_ppocr(image_id, bgr, pil, ground_truth):

    # Step 1: Import dependencies
    from paddleocr import PaddleOCR           # pip install paddlepaddle paddleocr

    # Step 2: Insert dataset image into the model
    ocr = PaddleOCR(use_angle_cls=False, lang="ta", show_log=False, use_gpu=False)
    result = ocr.ocr(bgr, cls=False)
    predicted = ""
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                predicted += line[1][0]

    # Step 3: Output accuracy compared to ground truth
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — Donut (Document Understanding Transformer)
# ─────────────────────────────────────────────────────────────────────────────

def model_4_donut(image_id, bgr, pil, ground_truth):

    # Step 1: Import dependencies
    from transformers import pipeline         # pip install transformers torch

    # Step 2: Insert dataset image into the model
    pipe = pipeline(
        "image-to-text",
        model="naver-clova-ix/donut-base-finetuned-cord-v2",
        trust_remote_code=True,
    )
    raw = pipe(pil)[0]["generated_text"]
    text = re.sub(r"<[^>]+>", " ", raw).strip()
    predicted = "".join(ch for ch in text if "\u0B80" <= ch <= "\u0BFF" or ch == " ").strip()

    # Step 3: Output accuracy compared to ground truth
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 5 — Perplexity LLM (Pixtral-12B multimodal)
# ─────────────────────────────────────────────────────────────────────────────

def model_5_perplexity(image_id, bgr, pil, ground_truth):

    # Step 1: Import dependencies
    from transformers import pipeline         # pip install transformers torch

    # Step 2: Insert dataset image into the model
    pipe = pipeline(
        "image-text-to-text",
        model="mistralai/Pixtral-12B-2409",
        trust_remote_code=True,
    )
    result = pipe(pil, max_new_tokens=256)
    raw = result[0]["generated_text"] if result else ""
    predicted = "".join(ch for ch in raw if "\u0B80" <= ch <= "\u0BFF" or ch == " ").strip()

    # Step 3: Output accuracy compared to ground truth
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 6 — DeepSeek LLM (DeepSeek-OCR)
# ─────────────────────────────────────────────────────────────────────────────

def model_6_deepseek(image_id, bgr, pil, ground_truth):

    # Step 1: Import dependencies
    from transformers import pipeline         # pip install transformers torch

    # Step 2: Insert dataset image into the model
    pipe = pipeline(
        "image-text-to-text",
        model="deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
    )
    result = pipe(pil, max_new_tokens=512)
    raw = result[0]["generated_text"] if result else ""
    predicted = "".join(ch for ch in raw if "\u0B80" <= ch <= "\u0BFF" or ch == " ").strip()

    # Step 3: Output accuracy compared to ground truth
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    ("Palm Leaf CNN (Baseline)",          model_1_cnn),
    ("PARSeq",                            model_2_parseq),
    ("PP-OCRv5 (ta_PP-OCRv5_mobile_rec)", model_3_ppocr),
    ("Donut (Document VLM)",              model_4_donut),
    ("Perplexity LLM (Pixtral-12B)",      model_5_perplexity),
    ("DeepSeek LLM (DeepSeek-OCR)",       model_6_deepseek),
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_all(max_images=5):
    ground_truths = load_ground_truth()
    image_ids     = sorted(ground_truths.keys())[:max_images]
    all_results   = []

    for image_id in image_ids:
        gt = ground_truths[image_id]
        print(f"\n{'─'*60}")
        print(f"  Image : {image_id}  |  GT chars : {len(gt)}")
        print(f"{'─'*60}")

        try:
            bgr, pil = load_image(image_id)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        for model_name, model_fn in MODELS:
            t0 = time.perf_counter()
            try:
                predicted, acc, cer = model_fn(image_id, bgr, pil, gt)
                status = "OK"
            except Exception as e:
                predicted, acc, cer = "", 0.0, 1.0
                status = f"SKIPPED: {e}"
            elapsed = round(time.perf_counter() - t0, 2)

            bar = "#" * int(acc / 5) if status == "OK" else ""
            tag = f"Acc={acc:5.1f}%  [{bar:<20}]" if status == "OK" else "SKIPPED"
            print(f"  {model_name:<38}  {tag}  ({elapsed}s)")

            all_results.append({
                "model":      model_name,
                "image_id":   image_id,
                "gt_chars":   len(gt),
                "pred_chars": len(predicted),
                "cer_%":      round(cer * 100, 2),
                "accuracy_%": acc,
                "time_sec":   elapsed,
                "status":     status,
            })

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results):
    model_rows = {}
    for r in all_results:
        model_rows.setdefault(r["model"], []).append(r)

    summary = []
    for name, rows in model_rows.items():
        valid = [r for r in rows if r["status"] == "OK"]
        n = len(valid)
        if n:
            avg_cer = round(sum(r["cer_%"] for r in valid) / n, 2)
            avg_acc = round(sum(r["accuracy_%"] for r in valid) / n, 2)
            avg_t   = round(sum(r["time_sec"] for r in valid) / n, 2)
        else:
            avg_cer, avg_acc, avg_t = "-", "-", "-"
        summary.append({"model": name, "tested": n,
                         "avg_cer": avg_cer, "avg_acc": avg_acc, "avg_t": avg_t})

    summary.sort(key=lambda x: x["avg_acc"] if isinstance(x["avg_acc"], float) else -1,
                 reverse=True)

    print(f"\n{'='*72}")
    print("  FINAL COMPARISON TABLE — Tamil Palm Leaf OCR")
    print(f"{'='*72}")
    print(f"  {'Rank':<5} {'Model':<40} {'CER%':>6} {'Acc%':>7} {'Time':>7}")
    print(f"  {'-'*65}")
    ranks = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}
    for i, row in enumerate(summary, 1):
        r = ranks.get(i, f"[{i:>3}]")
        cer = f"{row['avg_cer']:>6}" if isinstance(row['avg_cer'], float) else "     -"
        acc = f"{row['avg_acc']:>7}" if isinstance(row['avg_acc'], float) else "      -"
        t   = f"{row['avg_t']:>7}" if isinstance(row['avg_t'], float) else "      -"
        print(f"  {r:<5} {row['model']:<40} {cer} {acc} {t}")
    print(f"  {'-'*65}")
    print("  CER% = Character Error Rate (lower is better)")
    print("  Acc% = Character Accuracy   (higher is better)\n")
    return summary


def save_csv(all_results, summary):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = OUT_DIR / "comparison_per_image.csv"
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader(); w.writerows(all_results)
    print(f"[Saved] {p1}")

    p2 = OUT_DIR / "comparison_summary.csv"
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank","model","tested","avg_cer","avg_acc","avg_t"])
        w.writeheader()
        for i, row in enumerate(summary, 1):
            w.writerow({"rank": i, **row})
    print(f"[Saved] {p2}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=5,
                        help="Number of images to test (1-10, default=5)")
    args = parser.parse_args()

    print("=" * 72)
    print("  OCR MODEL COMPARISON — Tamil Palm Leaf Manuscript")
    print("=" * 72)

    results = run_all(max_images=min(max(1, args.images), 10))
    summary = print_summary(results)
    if results:
        save_csv(results, summary)
    print(f"[Done] Output -> {OUT_DIR}")
