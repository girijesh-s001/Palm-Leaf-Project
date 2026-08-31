import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.utils import to_categorical

from dataset_loader import load_dataset
from cnn_model import build_cnn_model


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("Training CNN model on palm leaf character dataset...")

    X_train, X_test, y_train, y_test, label_encoder = load_dataset(
        dataset_dir='dataset',
        test_size=0.2,
        random_state=42
    )

    num_classes = len(label_encoder.classes_)
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=num_classes)
    model.summary()

    epochs = 30
    batch_size = 32

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    summary_text = (
        f"\nEvaluation Results:\n"
        f"  Accuracy : {acc * 100:.2f}%\n"
        f"  Precision: {prec * 100:.2f}%\n"
        f"  Recall   : {rec * 100:.2f}%\n"
        f"  F1 Score : {f1 * 100:.2f}%\n"
    )
    print(summary_text)

    cls_report = classification_report(
        y_test, y_pred, labels=np.arange(num_classes), target_names=label_encoder.classes_, zero_division=0
    )

    os.makedirs('results', exist_ok=True)
    with open('results/metrics_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Palm Leaf Character Recognition - Metrics\n")
        f.write(summary_text)
        f.write("\nClassification Report:\n")
        f.write(cls_report)

    # Plot Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/accuracy_curve.png', dpi=300)
    plt.close()

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='#d62728', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#2ca02c', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/loss_curve.png', dpi=300)
    plt.close()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.close()

    # Save Model Artifacts
    model.save('cnn_model.h5')
    model.save('cnn_model.keras')
    model.save_weights('cnn_model.weights.h5')
    np.save('classes.npy', label_encoder.classes_)

    print("Model and evaluation results saved to 'results/' and root directory.")


if __name__ == '__main__':
    main()

