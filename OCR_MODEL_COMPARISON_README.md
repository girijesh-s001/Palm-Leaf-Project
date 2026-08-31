# OCR Model Comparison - Tamil Palm Leaf Manuscripts

Benchmark and architectural analysis comparing multiple OCR paradigms on the annotated Tamil Palm Leaf Manuscript dataset.

---

## Dataset & Ground Truth

| Metric | Detail |
|---|---|
| **Images** | `yaml annotation/dataset/images/img001.jpg` ... `img010.jpg` |
| **Annotations** | `yaml annotation/dataset/annotations/img001.yaml` ... `img010.yaml` |
| **Total Manuscripts** | 10 high-resolution palm-leaf photographs |
| **Total Labeled Glyphs** | ~2,200 handwritten Tamil character annotations |
| **Annotation Format** | Bounding box `[x, y, w, h]` with Unicode class label `["ன"]` |

Ground-truth text is assembled in spatial reading order (left-to-right, top-to-bottom) from the verified YAML annotation files.

---

## Evaluation Metric

* **Character Error Rate (CER):** Normalized Levenshtein edit distance between the predicted text string (H) and ground-truth text (R):
  $$\text{CER} = \frac{\text{EditDistance}(H, R)}{\text{Length}(R)}$$
* **Character Accuracy:**
  $$\text{Accuracy (\%)} = (1 - \text{CER}) \times 100$$

---

## Evaluated Models

The benchmark evaluates six OCR paradigms:

1. **Palm Leaf CNN (Baseline):** The project's dedicated segmentation and CNN pipeline trained directly on the manuscript dataset.
2. **PP-OCRv5 (Tamil):** PaddleOCR mobile recognition model (`ta_PP-OCRv5_mobile_rec`) pretrained on modern Tamil text.
3. **DeepSeek-OCR:** Multimodal vision-language model trained for document recognition.
4. **Pixtral-12B:** Multimodal large language model evaluated zero-shot.
5. **Donut:** End-to-end vision transformer for document visual question answering and transcription.
6. **PARSeq:** Permutation autoregressive sequence recognition model.

---

## Benchmark Summary

| Rank | Model | Paradigm | Tamil Support | Cost |
|:---:|---|---|:---:|:---:|
| **#1** | **Palm Leaf CNN (Baseline)** | Custom Segmentation + CNN | Trained on Palm Leaf Data | Free / Local |
| **#2** | **PP-OCRv5** | PaddleOCR (`lang="ta"`) | Pretrained Modern Tamil | Free / Local |
| **#3** | **DeepSeek-OCR** | Vision-Language Model | Multilingual OCR | Free / Local |
| **#4** | **Pixtral-12B** | Multimodal LLM | Multilingual Zero-Shot | API / Local |
| **#5** | **Donut** | Document VLM | Document Parsing | Free / Local |
| **#6** | **PARSeq** | Transformer Sequence Model | Requires Tamil Fine-Tuning | Free / Local |

---

## Running the Benchmark

```bash
# Benchmark on 5 sample images
python ocr_model_comparison.py --images 5

# Benchmark across all 10 annotated manuscripts
python ocr_model_comparison.py --images 10
```

### Outputs Generated
Output summaries and detailed per-image metrics are written to `output/comparison/`:
* `output/comparison/comparison_summary.csv`
* `output/comparison/comparison_per_image.csv`
