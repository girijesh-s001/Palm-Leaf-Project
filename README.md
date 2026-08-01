# 🌿 Palm Leaf OCR — Ancient Manuscript Character Recognition

> **What this project does in one sentence:**  
> Upload a photo of an ancient palm leaf manuscript → the system automatically reads it and converts the handwritten characters into digital text.

---

## 📖 Table of Contents

1. [What is this project?](#-what-is-this-project)
2. [How does it work? (Simple Explanation)](#-how-does-it-work-simple-explanation)
3. [Project Structure — Every File Explained](#-project-structure--every-file-explained)
4. [How to Install and Run](#-how-to-install-and-run)
5. [How to Use the Web Interface](#-how-to-use-the-web-interface)
6. [Understanding the Output](#-understanding-the-output)
7. [Technical Pipeline (For Developers)](#-technical-pipeline-for-developers)
8. [Model Accuracy](#-model-accuracy)
9. [Requirements](#-requirements)

---

## 🌴 What is this project?

Palm leaf manuscripts are ancient books written on dried palm leaves. They contain important historical, religious, and literary texts — but they are difficult to read digitally because:
- They are **handwritten** (not typed)
- The writing style is very different from modern fonts
- There is no automatic tool to read them

**This project solves that problem** by building an AI system that:
1. Takes a **photograph** of a palm leaf manuscript
2. Automatically finds each **line of text**
3. Cuts out each individual **character**
4. Uses a **trained AI model (CNN)** to recognize what each character is
5. Outputs the **full text** in correct reading order

---

## 🔍 How does it work? (Simple Explanation)

Think of it like this — imagine you are teaching someone to read a page:

```
Step 1: Clean the image         →  Remove dirt, shadows, noise from the photo
Step 2: Find the lines          →  Draw lines between each row of text
Step 3: Find each character     →  Draw a box around each individual letter/symbol
Step 4: Recognize the character →  AI model looks at each box and guesses what it is
Step 5: Combine everything      →  Join all recognized characters into full text
```

Here is the full flow as a diagram:

```
[Palm Leaf Photo]
       |
  [Image Cleaning]  <-- Remove noise, sharpen, convert to black & white
       |
  [Line Detection]  <-- Find where each row of text starts and ends
       |
[Character Cutting] <-- Isolate every individual character
       |
 [AI Recognition]   <-- CNN model identifies each character
       |
  [Final Text]      <-- Characters joined in reading order (left to right, top to bottom)
```

---

## 📁 Project Structure — Every File Explained

```
Palm leaf project/
|
|-- app.py                  <-- The web server (start this to use the browser interface)
|-- ocr_pipeline.py         <-- The main brain — connects all steps together
|-- README.md               <-- This file
|
|-- templates/
|   |-- index.html          <-- The web page you see in your browser
|
|-- LINE_SEG/               <-- The module that handles line & character detection
|   |-- line_seg.py         <-- Detects and separates lines of text
|   |-- char_seg.py         <-- Cuts each line into individual characters
|   |-- requirements.txt    <-- Python libraries needed for LINE_SEG
|   |-- input-6.jpeg        <-- Sample palm leaf test image
|   |-- input-7.jpeg        <-- Sample palm leaf test image
|   |-- output/             <-- Saved output images from segmentation
|   |-- sample/             <-- Sample input images for testing
|   |-- models/             <-- Saved model files (if any)
|   |-- test_verify.py      <-- Script to verify the segmentation output
|
|-- yaml annotation/        <-- The AI model training module
    |-- cnn_model.py        <-- Defines the AI model architecture
    |-- cnn_model.h5        <-- The pre-trained AI model file (ready to use)
    |-- train.py            <-- Script used to train the model (already done)
    |-- predict.py          <-- Loads model and predicts a character
    |-- preprocess.py       <-- Prepares an image before giving it to the AI
    |-- dataset_loader.py   <-- Loads training data during model training
    |-- classes.npy         <-- List of all characters the model can recognize
    |-- dataset/            <-- Training images of individual characters
    |-- accuracy_curve.png  <-- Graph showing model accuracy during training
    |-- loss_curve.png      <-- Graph showing model error during training
    |-- confusion_matrix.png<-- Chart showing which characters get confused
    |-- requirements.txt    <-- Python libraries needed for AI training
```

---

### Detailed File Descriptions

#### `app.py` — Web Server
This is the **entry point** for the web application. When you run `python app.py`, it starts a local web server at `http://localhost:5000`. It:
- Serves the web page (`index.html`) to your browser
- Receives an uploaded image from the browser
- Calls `ocr_pipeline.py` to process it
- Sends the results (text, images, accuracy) back to the browser

#### `ocr_pipeline.py` — The Main Pipeline
This is the **central coordinator** that connects all the pieces. When given an image, it:
1. Loads and cleans the image
2. Calls `line_seg.py` functions to find lines
3. Calls `char_seg.py` to find individual characters
4. Runs each character through the CNN model for recognition
5. Sorts everything in reading order (top to bottom, left to right)
6. Returns the complete recognized text and visual results

#### `templates/index.html` — The Web Interface
The **user-facing web page** built with HTML, CSS, and JavaScript. It provides:
- A drag-and-drop image upload box
- A preview of the selected image before processing
- A "Run Recognition" button
- Visual output showing detected lines and character boxes
- A text area showing the recognized characters
- A line-by-line breakdown table
- Accuracy badges (segmentation score and CNN model accuracy)

---

### `LINE_SEG/` Folder — Image Segmentation Module

#### `line_seg.py` — Line Segmentation
This file is responsible for **splitting the palm leaf image into individual rows of text**. It contains these main functions:

| Function | What it does |
|----------|-------------|
| `preprocess()` | Cleans the image — sharpens it, removes background noise, converts to binary (black and white) |
| `estimate_height()` | Measures the average height of characters to calibrate the line detection |
| `detect_separators()` | Finds the empty horizontal gaps between lines of text |
| `dp_trace()` | Draws a smooth curved path (separator line) between two rows of text, avoiding cutting through characters |
| `process_image()` | Runs the full pipeline on a single image when used as a standalone script |

#### `char_seg.py` — Character Segmentation
This file **splits each line of text into individual characters**. It contains:

| Function | What it does |
|----------|-------------|
| `get_initial_blocks()` | Scans each line from left to right using a column-by-column projection to find where characters start and end |
| `merge_overlapping_boxes()` | If two character boxes accidentally overlap, it merges them into one |
| `adaptive_split_and_save()` | The main function — processes each block, uses contour detection to separate touching characters, flags merged characters (two characters stuck together), and returns measurements for all characters |

#### `test_verify.py` — Verification Script
A small script to check if the segmentation output is correct by examining saved character images.

---

### `yaml annotation/` Folder — AI Model Module

#### `cnn_model.py` — Model Architecture
Defines the **Convolutional Neural Network (CNN)** structure — the layers, filters, and connections that make up the AI brain. Think of it like a blueprint for a building.

#### `cnn_model.h5` — Trained Model File
This is the **actual trained AI model** saved to disk. It contains everything the model learned during training. The pipeline loads this file to make predictions. (~26 MB)

#### `train.py` — Training Script
Used **once** to teach the model from the dataset. You do NOT need to run this again unless you want to retrain with new data.

#### `predict.py` — Prediction Module
Contains the `load_trained_model()` function that loads `cnn_model.h5` and provides a `predict()` function. The pipeline calls this for every character crop.

#### `preprocess.py` — Image Preprocessing for CNN
Before feeding a character image to the CNN, this script resizes it to **64x64 pixels**, converts it to grayscale, normalizes pixel values, and optionally inverts colors. This ensures every character is in a consistent format that the model expects.

#### `dataset_loader.py` — Training Data Loader
Reads character images from the `dataset/` folder and organizes them into training and testing sets. Used only during training.

#### `classes.npy` — Character Labels
A NumPy file containing the list of all character classes the model was trained to recognize. When the model outputs a prediction number (e.g., `42`), this file is used to look up the actual character name.

#### `accuracy_curve.png` — Training Accuracy Graph
A chart showing how the model's accuracy improved over each training epoch (cycle). Useful to see if training was successful.

#### `loss_curve.png` — Training Loss Graph
A chart showing how the model's error decreased during training. Lower is better.

#### `confusion_matrix.png` — Character Confusion Chart
A grid showing which characters the model sometimes confuses with each other.

---

## ⚙️ How to Install and Run

### Step 1 — Install Python
Make sure you have **Python 3.11** or higher installed.  
Download from: https://www.python.org/downloads/

### Step 2 — Install Required Libraries

Open a terminal/command prompt in the project folder and run:

```bash
pip install flask opencv-python numpy scipy tensorflow
```

Or install from the requirements files:

```bash
pip install -r "yaml annotation/requirements.txt"
pip install -r LINE_SEG/requirements.txt
pip install flask
```

### Step 3 — Start the Web Server

```bash
python app.py
```

You will see:

```
=======================================================
 Palm Leaf OCR - Web Interface
 Open your browser at: http://localhost:5000
=======================================================
```

### Step 4 — Open the Browser

Go to: **http://localhost:5000**

---

## 🖥️ How to Use the Web Interface

1. **Click the upload box** (or drag and drop an image onto it)
2. **Select a palm leaf image** (PNG, JPG, JPEG supported)
3. **Verify the preview** — make sure the correct image loaded
4. **Click "⚡ Run Recognition"**
5. **Wait a few seconds** — the system is:
   - Cleaning the image
   - Detecting lines
   - Cutting characters
   - Running the AI model on each character
6. **View the results:**
   - Line detection image (red lines drawn between rows)
   - Character boxes image (boxes drawn around each character)
   - Recognized text in the text area
   - Line-by-line breakdown table

---

## 📊 Understanding the Output

### Visual Outputs

| Output | What you see | What it means |
|--------|-------------|---------------|
| **Line Detection** | Original image with red curved lines | The system drew separator lines between each row of text |
| **Character Boxes** | White image with colored rectangles | Blue boxes = normal characters; Red boxes = merged or split characters |

### Text Output
The recognized characters are displayed in the text area in **exact reading order**: top line first, then second line, and within each line, left to right. You can click **"📋 Copy Text"** to copy it.

### Accuracy Badges

| Badge | Meaning |
|-------|---------|
| **X lines** | How many rows of text were detected |
| **X chars** | Total number of individual characters recognized |
| **CNN Model Accuracy: 89.19%** | The AI model is correct 89.19% of the time on its training/testing dataset |
| **Doc Segmentation: X%** | How cleanly the characters were separated in this specific image |

**Segmentation Score Color:**
- 🟢 **Green (80% and above)** — Excellent segmentation, clean image
- 🟡 **Yellow (55% to 79%)** — Acceptable, some merged characters
- 🔴 **Red (below 55%)** — Poor segmentation, image may be noisy or blurry

---

## 🔧 Technical Pipeline (For Developers)

### End-to-End Flow

```
run_ocr(image_path)                          [ocr_pipeline.py]
    |
    |-- preprocess(img)                       [line_seg.py]
    |      |-- Sharpen -> Threshold -> Denoise
    |      |-- Returns: cleaned binary, thresh_inv
    |
    |-- estimate_height(cleaned)              [line_seg.py]
    |      |-- Median height of connected components
    |
    |-- detect_separators(cleaned, text_h)    [line_seg.py]
    |      |-- Horizontal projection -> Gaussian smooth -> Valley detection
    |
    |-- dp_trace(reinforced, y)               [line_seg.py] (per separator)
    |      |-- Dynamic programming path avoiding text pixels
    |
    |-- [For each line strip]:
    |      |-- get_initial_blocks(line_bin)   [char_seg.py]
    |      |      |-- Vertical projection -> Left-right scan -> Raw blocks
    |      |-- adaptive_split_and_save(...)   [char_seg.py]
    |             |-- Contours -> Merge overlaps -> Flag merged chars
    |
    |-- Two-Stage Split (joined characters)   [ocr_pipeline.py]
    |      |-- Stage 1: Flag chars > 1.6x average width/height
    |      |-- Stage 2: Split at vertical histogram valley in center 30%
    |
    |-- Sort by (line_index, x_position)      [Reading order]
    |
    |-- _predict_crop(crop, model, classes)   [ocr_pipeline.py] (per char)
    |      |-- preprocess_pipeline -> 64x64 grayscale normalized tensor
    |      |-- model.predict() -> argmax -> character label + confidence
    |
    |-- Assemble text, accuracy metrics, visualizations
    |-- Return result dictionary
```

### Key Design Decisions

- **Dynamic Programming line tracing:** Instead of straight horizontal lines, the separator paths follow the contour of the writing using DP to avoid cutting through characters.
- **Two-stage character splitting:** A joined/merged character (two touching characters) is detected using aspect ratio + histogram valley analysis, then split at the lowest ink column in the middle 30% of the block.
- **Per-line statistics:** Average character size is computed per line (not just globally) to handle variation in handwriting size across the manuscript.
- **Source image for crops:** Character crops are extracted from the raw threshold image (preserving original ink detail), while segmentation logic uses the cleaned binary image.

---

## 📈 Model Accuracy

| Metric | Value |
|--------|-------|
| CNN Training / Dataset Accuracy | **89.19%** |
| CNN Test Accuracy | **48.30%** |
| Model Architecture | Convolutional Neural Network (CNN) |
| Input Size | 64 x 64 pixels (grayscale) |
| Model File | `yaml annotation/cnn_model.h5` (~26 MB) |

> **Note:** The gap between dataset accuracy (89%) and test accuracy (48%) indicates the model has learned the training data well, but real-world palm leaf images with variation in style, ink quality, and background are more challenging. This is an active area for improvement.

---

## 📦 Requirements

### Python Version
- **Python 3.11.9** (recommended)

### Libraries

| Library | Purpose |
|---------|---------|
| `flask` | Web server framework |
| `opencv-python` | Image processing (reading, thresholding, contours) |
| `numpy` | Numerical arrays and matrix operations |
| `scipy` | Signal processing (Gaussian smoothing, peak detection for line separators) |
| `tensorflow` | Loading and running the CNN model |

### Install All at Once

```bash
pip install flask opencv-python numpy scipy tensorflow
```

---

## 🚀 Quick Start Summary

```bash
# 1. Clone or download the project

# 2. Install dependencies
pip install flask opencv-python numpy scipy tensorflow

# 3. Start the server
python app.py

# 4. Open browser at http://localhost:5000

# 5. Upload a palm leaf image and click Run Recognition
```

---

*Built for digitizing and preserving ancient Tamil palm leaf manuscripts using computer vision and deep learning.*