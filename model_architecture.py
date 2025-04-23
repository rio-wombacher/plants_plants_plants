import tensorflow as tf
from tensorflow.keras import layers, models


def build_hybrid_model(input_shape=(256, 256, 3),binary=True):
    """
    Hybrid CAE and CNN model for image classification of if a plant is diseased and what type of disease"""
    inputs = tf.keras.Input(shape=input_shape)

    # CAE Encoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)  # Conv #1
    x = layers.MaxPooling2D((2, 2))(x)                                        # MaxPool #1

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)       # Conv #2
    x = layers.MaxPooling2D((2, 2))(x)                                        # MaxPool #2

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)       # Conv #3
    x = layers.MaxPooling2D((2, 2))(x)                                        # MaxPool #3

    # Bottleneck Layer
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)       # Bottleneck

    # CNN Classifier
    x = layers.Conv2D(6, (3, 3), activation='relu', padding='valid')(x)      # Conv #7
    x = layers.MaxPooling2D((2, 2))(x)                                        # MaxPool #4

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='valid')(x)     # Conv #8
    x = layers.MaxPooling2D((2, 2))(x)                                        # MaxPool #5

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='valid')(x)     # Conv #9
    x = layers.MaxPooling2D((2, 2))(x)                                        # MaxPool #6

    x = layers.Flatten()(x)                                                  # Flatten Layer
    x = layers.Dense(32, activation='relu')(x)                               # Dense #1

    if binary:
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)            # Dense #2
    else:
        outputs = tf.keras.layers.Dense(33, activation='softmax')(x) 

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

