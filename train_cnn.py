import tensorflow as tf
import matplotlib.pyplot as plt

from final_preprocessing import (
    train_dataset,
    validation_dataset,
    test_dataset,
    CLASSES
)

# =====================================
# 1. CONFIGURATION
# =====================================

EPOCHS = 30
MODEL_NAME = "veloria_cnn.keras"

# =====================================
# 2. DATA AUGMENTATION
# =====================================

augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomRotation(
        factor=0.015
    ),

    tf.keras.layers.RandomZoom(
        height_factor=0.10,
        width_factor=0.10
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    tf.keras.layers.RandomBrightness(
        factor=0.08,
        value_range=(0, 1)
    )

], name="data_augmentation")


# =====================================
# 3. BUILD CNN MODEL
# =====================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = augmentation(inputs)

# First convolution block
x = tf.keras.layers.Conv2D(
    32, (3, 3),
    activation="relu",
    padding="same"
)(x)

x = tf.keras.layers.MaxPooling2D()(x)

# Second convolution block
x = tf.keras.layers.Conv2D(
    64, (3, 3),
    activation="relu",
    padding="same"
)(x)

x = tf.keras.layers.MaxPooling2D()(x)

# Third convolution block
x = tf.keras.layers.Conv2D(
    128, (3, 3),
    activation="relu",
    padding="same"
)(x)

x = tf.keras.layers.MaxPooling2D()(x)

# Feature extraction
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Fully connected layer
x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.5)(x)

# Output layer
outputs = tf.keras.layers.Dense(
    len(CLASSES),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# =====================================
# 4. COMPILE MODEL
# =====================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()


# =====================================
# 5. CALLBACKS
# =====================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True

)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.00001

)

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_NAME,

    monitor="val_loss",

    save_best_only=True

)


# =====================================
# 6. TRAIN MODEL
# =====================================

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]

)


# =====================================
# 7. SAVE TRAINING HISTORY
# =====================================

model.save(MODEL_NAME)

print("\nModel saved successfully!")


# =====================================
# 8. PLOT TRAINING RESULTS
# =====================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    "cnn_training_history.png",
    dpi=150
)

plt.show()


print("\nCNN training completed!")