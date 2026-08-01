# CNN Baseline for Palm Leaf Character Recognition using Otsu Thresholding

This repository provides a baseline Convolutional Neural Network (CNN) model to evaluate the quality of the annotated Palm Leaf Character Dataset.

The objective of this experiment is **not to achieve state-of-the-art recognition accuracy**, but to verify that the collected dataset is suitable for deep learning based character recognition.

Every image is first converted into a binary image using **Otsu Thresholding**, followed by CNN training and evaluation.

---

# Workflow

```
                    Palm Leaf Dataset
                           │
                           ▼
              Read Image & YAML Annotation
                           │
                           ▼
                  Match Image with Label
                           │
                           ▼
                 Convert RGB → Grayscale
                           │
                           ▼
              Apply Otsu Thresholding
                           │
                           ▼
                   Binary Character Image
                           │
                           ▼
                    Resize to 64×64
                           │
                           ▼
                  Normalize Pixel Values
                           │
                           ▼
                 Encode Character Labels
                           │
                           ▼
                Train-Test Dataset Split
                     (80% / 20%)
                           │
                           ▼
                CNN Model Training
                           │
                           ▼
                 Evaluate Performance
                           │
        ┌──────────────────┼────────────────────┐
        ▼                  ▼                    ▼
   Accuracy          Confusion Matrix      Loss Curve
        │                  │                    │
        └──────────────────┼────────────────────┘
                           ▼
                Save Trained CNN Model
                           │
                           ▼
                Test New Character Image
                           │
                           ▼
              Predicted Character Label
```

---

# Dataset Structure

```
dataset/

│

├── images/
│      img001.jpg
│      img002.jpg
│      img003.jpg
│      ...
│
├── annotations/
│      img001.yaml
│      img002.yaml
│      img003.yaml
│      ...
```

Each YAML annotation contains `image_label` metadata and character bounding boxes with `glyph_id`, `bbox`, and `labels`.

---

# Image Preprocessing

Every image undergoes the following preprocessing pipeline.

```
RGB Image → Grayscale Conversion → Otsu Thresholding → Binary Image → Resize (64 × 64) → Normalization → CNN Input
```

Example

```
Original Image  ██████████  →  Otsu Threshold  ████░░████  →  CNN Input
```

---

# CNN Architecture

```
Input Image (64×64×1)
       │
       ▼
Conv2D (32 Filters, 3×3, ReLU)
       │
       ▼
MaxPooling2D (2×2)
       │
       ▼
Conv2D (64 Filters, 3×3, ReLU)
       │
       ▼
MaxPooling2D (2×2)
       │
       ▼
Conv2D (128 Filters, 3×3, ReLU)
       │
       ▼
MaxPooling2D (2×2)
       │
       ▼
Flatten
       │
       ▼
Dense (256 Units, ReLU)
       │
       ▼
Dropout (0.5)
       │
       ▼
Dense Softmax Layer (num_classes)
```

---

# Training Procedure

1. Read all YAML files and images from `dataset/`.
2. Match every annotation with its corresponding image.
3. Crop bounding boxes and convert images into binary using Otsu thresholding.
4. Resize character binary images to 64×64.
5. Normalize pixel values to `[0.0, 1.0]`.
6. Encode character labels using `LabelEncoder`.
7. Split dataset into Training (80%) and Testing (20%).
8. Train CNN for **30 Epochs**, **Batch Size 32**, **Adam Optimizer**, **Categorical Crossentropy Loss**.
9. Save trained model (`cnn_model.h5`).
10. Save label encoder (`classes.npy`).

---

# Prediction Workflow

```
Input Image → Grayscale → Otsu Threshold → Resize → Normalization → Load CNN Model → Predict Class → Display Character & Confidence Score
```

Example Output:
```
Input Image     : test.jpg
Predicted Label : அ
Confidence      : 98.64 %
```

---

# Installation

Clone repository:
```bash
git clone https://github.com/USERNAME/PalmLeafCNN.git
cd PalmLeafCNN
```

Create environment:
```bash
python -m venv venv
```

Activate environment:

- Windows:
```cmd
venv\Scripts\activate
```

- Linux / macOS:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

# Train Model

Execute training:
```bash
python train.py
```

Outputs generated:
- Accuracy, Precision, Recall, F1 Score printed to terminal
- Confusion Matrix (`confusion_matrix.png`)
- Accuracy Curve (`accuracy_curve.png`)
- Loss Curve (`loss_curve.png`)
- Saved CNN Model (`cnn_model.h5`)
- Label Encoder (`classes.npy`)

---

# Test Prediction

Execute character prediction:
```bash
python predict.py --image path/to/character.jpg
```

---

# Evaluation Metrics

✔ Accuracy  
✔ Precision  
✔ Recall  
✔ F1 Score  
✔ Confusion Matrix  
✔ Classification Report  
✔ Training Accuracy Curve  
✔ Validation Accuracy Curve  
✔ Training Loss Curve  
✔ Validation Loss Curve  

---

# Future Improvements

The CNN baseline can be extended using:
- Transfer Learning
- ResNet50
- EfficientNet
- Vision Transformer (ViT)
- CRNN
- DenseNet
- MobileNetV3
- Swin Transformer

---

# Research Objective

The objective of this baseline experiment is to determine whether the annotated Palm Leaf Character Dataset is suitable for deep learning based handwritten character recognition. This benchmark provides quantitative evidence of dataset quality before experimenting with advanced recognition models.


# Output sample
========================================
 Palm Leaf Character Prediction Result
========================================
Input Image     : test_sample.jpg
Predicted Label : ன
Confidence      : 96.51 %
========================================