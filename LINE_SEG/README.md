# Document Line & Character Segmentation Module

This module implements a rule-based image processing pipeline for segmenting scanned or photographed palm-leaf manuscripts into individual lines of text and isolated character bounding boxes. It is specifically designed for handwritten Indic scripts (such as Tamil), handling curved lines, overlapping strokes, and touching characters.

---

## Pipeline Overview

```
[Input Manuscript Image]
         │
         ▼
 1. Preprocessing & Denoising       (Sharpening, Adaptive Thresholding, Contour/CC Filtering)
         │
         ▼
 2. Line Segmentation              (CC Height Estimation & Dynamic Programming Seam Carving)
         │
         ▼
 3. Character Extraction           (Vertical Projection Profiles, Contour Bounding, Overlap Merge)
         │
         ▼
 4. Joint Character Resolution     (Aspect Ratio & Center Histogram Valley Splitting)
         │
         ▼
 5. Export & Metrics Output        (Clean Visualizations, Bounding Boxes, Segmentation Score)
```

---

## Technical Details

### 1. Preprocessing (`line_seg.py: preprocess`)
- **Sharpening:** Applies a 3x3 high-pass filter kernel to enhance faded character boundaries.
- **Adaptive Thresholding:** Binarizes the image using local mean thresholding ($21 \times 21$ window) to accommodate non-uniform illumination and background grain.
- **Dual Noise Filtering:**
  - *Contour filtering:* Removes outlier contours outside valid text bounds.
  - *Connected components (CC) filtering:* Strips small noise specks and large borders.

### 2. Line Segmentation (`line_seg.py: detect_separators` & `dp_trace`)
- **Baseline Scale Estimation:** Calculates median CC height to calibrate line spacing thresholds.
- **Valley Detection:** Locates inter-line troughs using a 1D Gaussian-smoothed horizontal ink profile.
- **Seam Carving (`dp_trace`):** Dynamic programming traces an energy-minimizing dividing line across the page from $x = 0$ to $x = W$.
  $$\text{Cost} = \text{DistanceTransformCost} + \text{TextCollisionPenalty} + \text{VerticalDeviationPenalty}$$
  The path avoids cutting through characters by navigating through maximum background clearance.

### 3. Character Segmentation (`char_seg.py: get_initial_blocks` & `adaptive_split_and_save`)
- **Initial Blocks:** Columns with zero text pixels in the vertical projection histogram define preliminary boundaries.
- **Contour Refinement:** Identifies independent disjoint components within each vertical strip.
- **Overlap Merging:** Iteratively unions 2D bounding boxes that overlap horizontally or vertically (e.g. vowel modifiers and diacritics).

### 4. Touching Character Splitting
Touching characters create unusually wide bounding boxes:
- Flags characters exceeding $1.6\times$ average width/height.
- Verifies aspect ratio ($> 1.6$) and checks for a prominent valley in the middle 30% width region.
- Splits verified joints at the column of minimum ink density into sub-parts `a` and `b`.

---

## Module Files

* **`line_seg.py`**: Core line detection, preprocessing, and standalone segmentation execution.
* **`char_seg.py`**: Character block scanning, 2D box merging, and adaptive contour refinement.
* **`requirements.txt`**: Python dependencies required for this module (`opencv-python`, `numpy`, `scipy`).

---

## Usage

### Run Standalone Visualizer

```bash
python LINE_SEG/line_seg.py path/to/manuscript.jpg
```

If run without arguments, a file selection dialog will appear. Outputs (line-segmented and character-segmented images) are written to the `output/` directory.
