import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from model_architecture import build_hybrid_model


def load_datasets(train_dir, val_dir, test_dir, target_size=(256, 256), batch_size=32):
    """
    Load train, validation, and test datasets using ImageDataGenerator
    """
    train_gen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

    val_test_gen = ImageDataGenerator(rescale=1.0/255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = val_test_gen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = val_test_gen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important for evaluation
    )

    return train_data, val_data, test_data


def train_and_evaluate():
    # Paths
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'
    model_path = 'best_model.keras'
    input_shape = (256, 256, 3)
    num_classes = 38
    batch_size = 32
    epochs = 30

    # Load datasets
    train_data, val_data, test_data = load_datasets(train_dir, val_dir, test_dir, target_size=input_shape[:2], batch_size=batch_size)

    # Build model
    model = build_hybrid_model(input_shape=input_shape, num_classes=num_classes)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1)
    early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', verbose=1)

    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    best_model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = best_model.evaluate(test_data)
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    train_and_evaluate()   