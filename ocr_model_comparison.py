import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE_DIR = Path(__file__).parent
ANN_DIR = BASE_DIR / "yaml annotation" / "dataset" / "annotations"
IMG_DIR = BASE_DIR / "yaml annotation" / "dataset" / "images"
OUT_DIR = BASE_DIR / "output" / "comparison"


def load_ground_truth():
    """Loads ground-truth character sequences from YAML annotations."""
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

    print(f"Loaded ground truth for {len(ground_truths)} images.")
    return ground_truths


def load_image(image_id):
    from PIL import Image as PILImage
    path = IMG_DIR / f"{image_id}.jpg"
    if not path.exists():
        path = IMG_DIR / f"{image_id}.png"
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image for ID: {image_id}")
    pil_img = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return bgr, pil_img


def compute_cer(predicted, ground_truth):
    """Computes Character Error Rate via Levenshtein distance."""
    if not ground_truth:
        return 0.0 if not predicted else 1.0

    ref, hyp = ground_truth, predicted
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return min(dp[n] / len(ref), 1.0)


# Model evaluators

def evaluate_palm_leaf_cnn(image_id, bgr, pil, ground_truth):
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from ocr_pipeline import run_ocr

    result = run_ocr(bgr)
    predicted = result.get("text", "")
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


def evaluate_parseq(image_id, bgr, pil, ground_truth):
    import torch

    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    model.eval()
    tensor = model.get_transform()(pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    pred, _ = model.tokenizer.decode(logits)
    predicted = pred[0] if pred else ""
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


def evaluate_paddle_ocr(image_id, bgr, pil, ground_truth):
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=False, lang="ta", show_log=False, use_gpu=False)
    result = ocr.ocr(bgr, cls=False)
    predicted = ""
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                predicted += line[1][0]
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


def evaluate_donut(image_id, bgr, pil, ground_truth):
    from transformers import pipeline

    pipe = pipeline("image-to-text", model="naver-clova-ix/donut-base-finetuned-cord-v2", trust_remote_code=True)
    raw = pipe(pil)[0]["generated_text"]
    text = re.sub(r"<[^>]+>", " ", raw).strip()
    predicted = "".join(ch for ch in text if "\u0B80" <= ch <= "\u0BFF" or ch == " ").strip()
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


def evaluate_pixtral(image_id, bgr, pil, ground_truth):
    from transformers import pipeline

    pipe = pipeline("image-text-to-text", model="mistralai/Pixtral-12B-2409", trust_remote_code=True)
    result = pipe(pil, max_new_tokens=256)
    raw = result[0]["generated_text"] if result else ""
    predicted = "".join(ch for ch in raw if "\u0B80" <= ch <= "\u0BFF" or ch == " ").strip()
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


def evaluate_deepseek_ocr(image_id, bgr, pil, ground_truth):
    from transformers import pipeline

    pipe = pipeline("image-text-to-text", model="deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
    result = pipe(pil, max_new_tokens=512)
    raw = result[0]["generated_text"] if result else ""
    predicted = "".join(ch for ch in raw if "\u0B80" <= ch <= "\u0BFF" or ch == " ").strip()
    cer = compute_cer(predicted, ground_truth)
    acc = round((1 - cer) * 100, 2)
    return predicted, acc, cer


MODELS = [
    ("Palm Leaf CNN (Baseline)", evaluate_palm_leaf_cnn),
    ("PARSeq", evaluate_parseq),
    ("PP-OCRv5 (Tamil)", evaluate_paddle_ocr),
    ("Donut (Document VLM)", evaluate_donut),
    ("Pixtral-12B (VLM)", evaluate_pixtral),
    ("DeepSeek-OCR", evaluate_deepseek_ocr),
]


def run_benchmark(max_images=5):
    ground_truths = load_ground_truth()
    image_ids = sorted(ground_truths.keys())[:max_images]
    all_results = []

    for image_id in image_ids:
        gt = ground_truths[image_id]
        print(f"\nProcessing Image: {image_id} (GT length: {len(gt)})")

        try:
            bgr, pil = load_image(image_id)
        except Exception as e:
            print(f"Error loading {image_id}: {e}")
            continue

        for model_name, eval_fn in MODELS:
            t0 = time.perf_counter()
            try:
                predicted, acc, cer = eval_fn(image_id, bgr, pil, gt)
                status = "OK"
            except Exception as e:
                predicted, acc, cer = "", 0.0, 1.0
                status = f"SKIPPED ({e})"

            elapsed = round(time.perf_counter() - t0, 2)
            print(f"  - {model_name:<30} Acc: {acc:>5.1f}% | CER: {cer * 100:>5.1f}% | Time: {elapsed:>5.2f}s | Status: {status}")

            all_results.append({
                "model": model_name,
                "image_id": image_id,
                "gt_chars": len(gt),
                "pred_chars": len(predicted),
                "cer_%": round(cer * 100, 2),
                "accuracy_%": acc,
                "time_sec": elapsed,
                "status": status,
            })

    return all_results


def print_and_save_summary(all_results):
    model_groups = {}
    for r in all_results:
        model_groups.setdefault(r["model"], []).append(r)

    summary = []
    for name, rows in model_groups.items():
        valid = [r for r in rows if r["status"] == "OK"]
        if valid:
            avg_cer = round(sum(r["cer_%"] for r in valid) / len(valid), 2)
            avg_acc = round(sum(r["accuracy_%"] for r in valid) / len(valid), 2)
            avg_t = round(sum(r["time_sec"] for r in valid) / len(valid), 2)
        else:
            avg_cer, avg_acc, avg_t = "-", "-", "-"
        summary.append({
            "model": name,
            "tested": len(valid),
            "avg_cer": avg_cer,
            "avg_acc": avg_acc,
            "avg_t": avg_t,
        })

    summary.sort(key=lambda x: x["avg_acc"] if isinstance(x["avg_acc"], float) else -1, reverse=True)

    print("\nBenchmark Summary:")
    print(f"{'Rank':<5} {'Model':<32} {'Avg CER%':>10} {'Avg Acc%':>10} {'Avg Time':>10}")
    print("-" * 72)
    for i, row in enumerate(summary, 1):
        cer = f"{row['avg_cer']}%" if isinstance(row['avg_cer'], (int, float)) else "-"
        acc = f"{row['avg_acc']}%" if isinstance(row['avg_acc'], (int, float)) else "-"
        t = f"{row['avg_t']}s" if isinstance(row['avg_t'], (int, float)) else "-"
        print(f"#{i:<4} {row['model']:<32} {cer:>10} {acc:>10} {t:>10}")
    print("-" * 72)

    # Save to CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if all_results:
        per_img_path = OUT_DIR / "comparison_per_image.csv"
        with open(per_img_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        summary_path = OUT_DIR / "comparison_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "model", "tested", "avg_cer", "avg_acc", "avg_t"])
            writer.writeheader()
            for i, row in enumerate(summary, 1):
                writer.writerow({"rank": i, **row})

        print(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate & compare OCR models on palm leaf dataset")
    parser.add_argument("--images", type=int, default=5, help="Number of images to benchmark (default: 5)")
    args = parser.parse_args()

    results = run_benchmark(max_images=max(1, args.images))
    print_and_save_summary(results)

