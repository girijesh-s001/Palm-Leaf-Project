import sys
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from preprocess import preprocess_pipeline
from cnn_model import build_cnn_model


def load_trained_model(classes_count: int, model_path: str = 'cnn_model.h5', weights_path: str = 'cnn_model.weights.h5'):
    """Loads the trained model, trying saved model files or rebuilding from weights."""
    keras_path = 'cnn_model.keras'
    if os.path.exists(keras_path):
        try:
            return tf.keras.models.load_model(keras_path)
        except Exception:
            pass

    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception:
            pass

    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=classes_count)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    elif os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        raise FileNotFoundError("Trained model or weights not found. Run train.py first.")

    return model


def predict_character(image_path: str, model_path: str = 'cnn_model.h5', classes_path: str = 'classes.npy'):
    """Predicts the character label for an input image crop."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file not found: {classes_path}")

    classes = np.load(classes_path, allow_pickle=True)
    model = load_trained_model(classes_count=len(classes), model_path=model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    processed = preprocess_pipeline(img, target_size=(64, 64), invert=True)
    input_tensor = np.expand_dims(processed, axis=0)

    predictions = model.predict(input_tensor, verbose=0)[0]
    idx = int(np.argmax(predictions))
    confidence = float(predictions[idx]) * 100.0
    label = str(classes[idx])

    print(f"Image     : {image_path}")
    print(f"Predicted : {label}")
    print(f"Confidence: {confidence:.2f}%")

    return label, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict palm leaf character from image")
    parser.add_argument("--image", type=str, required=True, help="Path to character crop image")
    args = parser.parse_args()

    predict_character(args.image)

