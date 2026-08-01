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

    print("=== Palm Leaf Character Recognition - CNN Baseline Training ===")

    # 1. Load and Preprocess Dataset (80% Train, 20% Test)
    dataset_dir = 'dataset'
    X_train, X_test, y_train, y_test, label_encoder = load_dataset(
        dataset_dir=dataset_dir,
        test_size=0.2,
        random_state=42
    )

    num_classes = len(label_encoder.classes_)
    print(f"Number of target character classes: {num_classes}")

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # 2. Build CNN Model
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=num_classes)
    model.summary()

    # 3. Train Model
    epochs = 30
    batch_size = 32
    print(f"\nStarting CNN Model Training for {epochs} Epochs (Batch Size: {batch_size})...")

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 4. Evaluate Performance on Test Set
    print("\n=== Evaluating Model Performance on Test Set ===")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    summary_text = (
        "\n--- Evaluation Metrics ---\n"
        f"[+] Accuracy : {acc * 100:.2f} %\n"
        f"[+] Precision: {prec * 100:.2f} %\n"
        f"[+] Recall   : {rec * 100:.2f} %\n"
        f"[+] F1 Score : {f1 * 100:.2f} %\n"
    )

    try:
        print(summary_text)
    except Exception:
        print(summary_text.encode('ascii', errors='replace').decode('ascii'))

    cls_report = classification_report(
        y_test, y_pred, labels=np.arange(num_classes), target_names=label_encoder.classes_, zero_division=0
    )

    # Write summary & classification report to file with UTF-8
    os.makedirs('results', exist_ok=True)
    with open('results/metrics_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=== Palm Leaf Character Recognition Results ===\n")
        f.write(summary_text)
        f.write("\n--- Classification Report ---\n")
        f.write(cls_report)

    # 5. Plot & Save Curves and Confusion Matrix
    # Plot Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    plt.title('Palm Leaf Character Recognition - Accuracy Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('accuracy_curve.png', dpi=300)
    plt.savefig('results/accuracy_curve.png', dpi=300)
    plt.close()
    print("[+] Accuracy curve saved to 'accuracy_curve.png'")

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='#d62728', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#2ca02c', linewidth=2)
    plt.title('Palm Leaf Character Recognition - Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Categorical Crossentropy Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    plt.savefig('results/loss_curve.png', dpi=300)
    plt.close()
    print("[+] Loss curve saved to 'loss_curve.png'")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.title('Palm Leaf Character Recognition - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label Index', fontsize=12)
    plt.ylabel('True Label Index', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.close()
    print("[+] Confusion matrix saved to 'confusion_matrix.png'")

    # 6. Save Trained Model and Label Encoder
    model.save('cnn_model.h5')
    model.save('cnn_model.keras')
    model.save_weights('cnn_model.weights.h5')
    np.save('classes.npy', label_encoder.classes_)

    print("\n[+] Model saved successfully to 'cnn_model.h5' and 'cnn_model.keras'")
    print("[+] Model weights saved to 'cnn_model.weights.h5'")
    print("[+] Label encoder classes saved successfully to 'classes.npy'")


if __name__ == '__main__':
    main()
