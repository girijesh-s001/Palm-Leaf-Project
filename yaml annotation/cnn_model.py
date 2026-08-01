import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(64, 64, 1), num_classes: int = 128) -> tf.keras.Model:
    """Builds the baseline CNN model for character recognition as per specification:

    Input (64x64x1) -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool ->
    Conv2D(128) -> MaxPool -> Flatten -> Dense(256) -> Dropout -> Softmax
    """
    model = models.Sequential([
        # Layer 1: Conv2D 32 filters + MaxPooling
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Layer 2: Conv2D 64 filters + MaxPooling
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Layer 3: Conv2D 128 filters + MaxPooling
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Flatten & Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        # Softmax Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=128)
    model.summary()
