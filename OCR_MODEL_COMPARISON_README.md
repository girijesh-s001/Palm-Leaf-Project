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

| Rank | Model | Paradigm | Character Accuracy (%) | End-to-End Accuracy (1-CER) | Tamil Support | Primary Drawback / Reason for Drawback | Cost / Runtime |
|:---:|---|---|:---:|:---:|:---:|---|:---:|
| **#1** | **Palm Leaf CNN (Baseline)** | Custom Segmentation + CNN | **48.30%** (F1: 46.17%) | **~10.6% – 24.5%** | Trained on Palm Leaf Data | High sensitivity to segmentation errors; struggles with severe class imbalance on rare ligatures and touching glyphs. | Free / Local (Fast) |
| **#2** | **PP-OCRv5** | PaddleOCR (`lang="ta"`) | **~5.2%** | **~2.8%** | Pretrained Modern Tamil | Trained strictly on clean modern printed fonts; cannot recognize ancient cursive palm-leaf glyphs or low-contrast incisions. | Free / Local (Moderate) |
| **#3** | **DeepSeek-OCR** | Vision-Language Model | **~1.8%** | **~0.9%** | Multilingual OCR | Zero-shot hallucination; generates modern conversational Tamil approximations rather than verbatim ancient character sequences. | Free / Local (Heavy) |
| **#4** | **Pixtral-12B** | Multimodal LLM | **~1.1%** | **~0.5%** | Multilingual Zero-Shot | Lacks fine-grained bounding box grounding; suffers from conversational drift and token truncation on dense manuscript lines. | API / Local (GPU Heavy) |
| **#5** | **Donut** | Document VLM | **0.0%** | **0.0%** | Document Parsing | Pretrained on Latin receipts/forms; Tamil Unicode characters are completely outside its tokenization vocabulary. | Free / Local (Moderate) |
| **#6** | **PARSeq** | Transformer Sequence Model | **0.0%** | **0.0%** | Requires Tamil Fine-Tuning | Vocabulary restricted to Latin alphanumeric characters `[0-9a-zA-Z]`; cannot output Tamil script without domain fine-tuning. | Free / Local (Fast) |



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
