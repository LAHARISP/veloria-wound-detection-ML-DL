import tensorflow as tf
import matplotlib.pyplot as plt

from final_preprocessing import (
    train_dataset,
    CLASSES
)

# Data augmentation layers
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

])

# Get one training batch
images, labels = next(iter(train_dataset))

# Apply augmentation
augmented_images = augmentation(
    images,
    training=True
)

# Display original and augmented images
plt.figure(figsize=(12, 10))

for i in range(8):

    # Original
    plt.subplot(8, 2, 2 * i + 1)

    plt.imshow(images[i].numpy())
    plt.title(
        f"Original: {CLASSES[labels[i].numpy()]}"
    )

    plt.axis("off")

    # Augmented
    plt.subplot(8, 2, 2 * i + 2)

    plt.imshow(
        tf.clip_by_value(
            augmented_images[i],
            0,
            1
        ).numpy()
    )

    plt.title("Augmented")
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "augmentation_comparison.png",
    dpi=150
)

plt.show()

print("Augmentation preview completed!")