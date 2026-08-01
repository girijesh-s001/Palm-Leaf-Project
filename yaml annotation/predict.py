import sys
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from preprocess import preprocess_pipeline
from cnn_model import build_cnn_model


def load_trained_model(classes_count: int, model_path: str = 'cnn_model.h5', weights_path: str = 'cnn_model.weights.h5'):
    """Loads the trained CNN model with fallbacks for cross-version Keras compatibility."""
    model = None

    # Priority 1: Try native .keras format if it exists
    keras_path = 'cnn_model.keras'
    if os.path.exists(keras_path):
        try:
            model = tf.keras.models.load_model(keras_path)
            return model
        except Exception:
            pass

    # Priority 2: Try .h5 format load_model
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception:
            pass

    # Priority 3: Rebuild architecture and load weights
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=classes_count)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    elif os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        raise FileNotFoundError("Could not locate trained model or weights file. Please run train.py first.")

    return model


def predict_character(image_path: str, model_path: str = 'cnn_model.h5', classes_path: str = 'classes.npy'):
    """Performs character recognition prediction on an input image.

    Pipeline:
    Input Image -> Grayscale -> Otsu Threshold -> Resize (64x64) -> Normalization -> CNN -> Predict
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file not found: {classes_path}. Please run train.py first.")

    # 1. Load Label Encoder Classes
    classes = np.load(classes_path, allow_pickle=True)

    # 2. Load Model cleanly
    model = load_trained_model(classes_count=len(classes), model_path=model_path)

    # 3. Read Image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 4. Preprocess Image
    processed_img = preprocess_pipeline(img, target_size=(64, 64), invert=True)

    # Add batch dimension (1, 64, 64, 1)
    input_tensor = np.expand_dims(processed_img, axis=0)

    # 5. Predict
    predictions = model.predict(input_tensor, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx] * 100.0
    predicted_label = classes[predicted_idx]

    # Display results
    print("\n" + "=" * 40)
    print(" Palm Leaf Character Prediction Result ")
    print("=" * 40)
    print(f"Input Image     : {image_path}")

    try:
        print(f"Predicted Label : {predicted_label}")
    except Exception:
        print(f"Predicted Label : {predicted_label.encode('utf-8')}")

    print(f"Confidence      : {confidence:.2f} %")
    print("=" * 40 + "\n")

    return predicted_label, confidence


def create_sample_test_image():
    """Generates a test character crop from the dataset if no input image is provided."""
    import yaml
    import re

    annotations_dir = 'dataset/annotations'
    images_dir = 'dataset/images'

    if not os.path.exists(annotations_dir) or not os.path.exists(images_dir):
        return None

    ann_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.yaml')])
    if not ann_files:
        return None

    ann_path = os.path.join(annotations_dir, ann_files[0])
    base_name = os.path.splitext(ann_files[0])[0]
    img_path = os.path.join(images_dir, base_name + '.jpg')

    with open(ann_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content_fixed = re.sub(r'(\n\s*labels:\s*)\n\[\]', r'\1 []', content)
    data = yaml.safe_load(content_fixed)

    anns = data.get('annotations', [])
    for a in anns:
        bbox = a.get('bbox')
        lbls = a.get('labels')
        if bbox and lbls and len(bbox) >= 4:
            x, y, w, h = [int(v) for v in bbox[:4]]
            img = cv2.imread(img_path)
            if img is not None:
                crop = img[y:y+h, x:x+w]
                test_path = 'test_sample.jpg'
                cv2.imwrite(test_path, crop)
                return test_path
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Palm Leaf Character Recognition Prediction")
    parser.add_argument("--image", type=str, help="Path to input character image for prediction")
    args = parser.parse_args()

    target_image = args.image

    if not target_image:
        print("No image provided via --image. Generating test character sample from dataset...")
        target_image = create_sample_test_image()

    if target_image and os.path.exists(target_image):
        predict_character(target_image)
    else:
        print("Error: No test image available for prediction.")
