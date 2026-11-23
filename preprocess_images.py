import tensorflow as tf
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "split_dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

CLASSES = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================
# 2. LOAD DATASETS
# ==========================================

print("\nLoading image datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=None,
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=None,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=None,
)


# ==========================================
# 3. NORMALIZATION
# ==========================================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


# ==========================================
# 4. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomRotation(0.05),

    tf.keras.layers.RandomZoom(0.10),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

], name="data_augmentation")


# ==========================================
# 5. PREPROCESS TRAINING DATA
# ==========================================

train_ds = train_ds.map(
    lambda images, labels: (
        data_augmentation(
            normalization_layer(images),
            training=True
        ),
        labels
    ),
    num_parallel_calls=AUTOTUNE
)


# ==========================================
# 6. PREPROCESS VALIDATION DATA
# ==========================================

validation_ds = validation_ds.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=AUTOTUNE
)


# ==========================================
# 7. PREPROCESS TEST DATA
# ==========================================

test_ds = test_ds.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=AUTOTUNE
)


# ==========================================
# 8. PERFORMANCE OPTIMIZATION
# ==========================================

train_ds = train_ds.prefetch(AUTOTUNE)

validation_ds = validation_ds.prefetch(AUTOTUNE)

test_ds = test_ds.prefetch(AUTOTUNE)


# ==========================================
# 9. VERIFY PREPROCESSED DATA
# ==========================================

print("\n===================================")
print("PREPROCESSING VERIFICATION")
print("===================================")

for images, labels in train_ds.take(1):

    print("Image batch shape:", images.shape)

    print("Label batch shape:", labels.shape)

    print("Minimum pixel value:",
          float(tf.reduce_min(images)))

    print("Maximum pixel value:",
          float(tf.reduce_max(images)))


print("\nClass names:", CLASSES)

print("\nTraining batches:",
      len(train_ds))

print("Validation batches:",
      len(validation_ds))

print("Testing batches:",
      len(test_ds))

print("\nImage preprocessing completed successfully!")