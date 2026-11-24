import tensorflow as tf
from pathlib import Path

# Configuration
BASE_DIR = Path("split_dataset")

CLASSES = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

IMAGE_SIZE = 224
BATCH_SIZE = 16

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}

# Load image paths and labels
def get_image_paths(split):

    paths = []
    labels = []

    for label, class_name in enumerate(CLASSES):

        folder = BASE_DIR / split / class_name

        for image_path in sorted(folder.iterdir()):

            if image_path.suffix.lower() in SUPPORTED_FORMATS:

                paths.append(str(image_path))
                labels.append(label)

    return paths, labels


# Image preprocessing function
def preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    # Preserve aspect ratio and add padding
    image = tf.image.resize_with_pad(
        image,
        IMAGE_SIZE,
        IMAGE_SIZE,
        method="bilinear"
    )

    # Normalize pixels
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# Create TensorFlow dataset
def create_dataset(split, shuffle=False):

    paths, labels = get_image_paths(split)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=42
        )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"{split.capitalize()} images: {len(paths)}")

    return dataset


# Load datasets
print("\nLoading datasets...\n")

train_dataset = create_dataset("train", shuffle=True)

validation_dataset = create_dataset("validation")

test_dataset = create_dataset("test")


# Verify preprocessing
images, labels = next(iter(train_dataset))

print("\nPREPROCESSING VERIFICATION")
print("=" * 40)

print("Image shape:", images.shape)
print("Label shape:", labels.shape)

print("Minimum pixel:", tf.reduce_min(images).numpy())
print("Maximum pixel:", tf.reduce_max(images).numpy())

print("\nClass names:", CLASSES)

print(
    "Training batches:",
    sum(1 for _ in train_dataset)
)

print(
    "Validation batches:",
    sum(1 for _ in validation_dataset)
)

print(
    "Testing batches:",
    sum(1 for _ in test_dataset)
)

print("\nFinal preprocessing completed!")