# CNN Character Recognition Module

This module defines, trains, and evaluates the Convolutional Neural Network (CNN) used for recognizing isolated Tamil characters extracted from palm-leaf manuscripts.

---

## Workflow

```
[Character Crop] ──► Preprocessing ──► Grayscale ──► Otsu Threshold ──► Resize 64x64 ──► Normalize [0,1]
                                                                                               │
                                                                                               ▼
[Predicted Class & Confidence] ◄── Argmax Lookup ◄── Softmax Output ◄── CNN Model (3 Conv/Pool)
```

---

## Module Files

| File | Purpose |
|---|---|
| **`cnn_model.py`** | Sequential Keras CNN architecture definition (3 Conv2D layers, MaxPool, Dense, Dropout). |
| **`dataset_loader.py`** | Reads images and YAML bounding-box annotations, extracts crops, and prepares stratified train/test splits. |
| **`preprocess.py`** | Standardization pipeline: grayscale conversion, Otsu thresholding, 64x64 resizing, and float normalization. |
| **`train.py`** | Model training script (30 epochs, batch size 32, Adam optimizer) generating performance metrics and plots. |
| **`predict.py`** | CLI inference script for evaluating a single character crop image. |
| **`classes.npy`** | NumPy array mapping integer prediction indices to character Unicode labels. |
| **`cnn_model.h5`** | Pre-trained model weights. |
| **`dataset/`** | Contains annotated ground-truth images (`dataset/images/`) and YAML metadata (`dataset/annotations/`). |
| **`results/`** | Training evaluation artifacts: `accuracy_curve.png`, `loss_curve.png`, `confusion_matrix.png`, and `metrics_summary.txt`. |

---

## CNN Architecture

```
Input: (64, 64, 1)
  ├── Conv2D(32, 3x3, ReLU, same padding) ──► MaxPooling2D(2x2)
  ├── Conv2D(64, 3x3, ReLU, same padding) ──► MaxPooling2D(2x2)
  ├── Conv2D(128, 3x3, ReLU, same padding) ──► MaxPooling2D(2x2)
  ├── Flatten
  ├── Dense(256, ReLU)
  ├── Dropout(0.5)
  └── Dense(num_classes, Softmax)
```

---

## Usage

### Train the Model

```bash
python "yaml annotation/train.py"
```

Generates updated weights (`cnn_model.h5`, `cnn_model.keras`), class label encodings (`classes.npy`), and evaluation graphs in `yaml annotation/results/`.

### Predict a Single Character

```bash
python "yaml annotation/predict.py" --image path/to/character_crop.png
```