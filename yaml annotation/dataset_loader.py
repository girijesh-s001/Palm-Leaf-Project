import os
import re
import cv2
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_pipeline


def load_fixed_yaml(filepath: str) -> dict:
    """Loads YAML annotations while handling potential formatting issues like unindented empty lists."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Fix unindented empty list after labels:
    content_fixed = re.sub(r'(\n\s*labels:\s*)\n\[\]', r'\1 []', content)
    return yaml.safe_load(content_fixed)


def load_dataset(
    dataset_dir: str = 'dataset',
    test_size: float = 0.2,
    random_state: int = 42
):
    """Loads images and YAML annotations from dataset_dir, crops character bounding boxes,

    preprocesses glyphs, encodes labels, and performs train-test split.

    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'annotations')

    X = []
    y = []

    ann_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.yaml')])

    for ann_file in ann_files:
        base_name = os.path.splitext(ann_file)[0]
        yaml_path = os.path.join(annotations_dir, ann_file)

        # Look for corresponding image file (.jpg, .jpeg, .png)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            possible = os.path.join(images_dir, base_name + ext)
            if os.path.exists(possible):
                img_path = possible
                break

        if img_path is None:
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        data = load_fixed_yaml(yaml_path)

        for ann in data.get('annotations', []):
            labels = ann.get('labels')
            bbox = ann.get('bbox')

            if not labels or len(labels) == 0:
                continue
            if not bbox or len(bbox) < 4:
                continue

            x, y_pos, w, h = [int(v) for v in bbox[:4]]
            if w <= 0 or h <= 0:
                continue

            x1, y1 = max(0, x), max(0, y_pos)
            x2, y2 = min(img_w, x + w), min(img_h, y_pos + h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            processed = preprocess_pipeline(crop, target_size=(64, 64), invert=True)

            X.append(processed)
            y.append(labels[0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"Total dataset samples: {len(X)}")

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Total unique character classes: {num_classes}")

    # Check class distribution for stratification safety
    unique_classes, counts = np.unique(y_encoded, return_counts=True)
    can_stratify = np.all(counts >= 2)

    stratify_arg = y_encoded if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
        shuffle=True
    )

    print(f"Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"Testing samples: {len(X_test)} ({test_size*100:.0f}%)")

    return X_train, X_test, y_train, y_test, label_encoder
