import os
import matplotlib.pyplot as plt
from PIL import Image
import random

dataset_path = "wound_dataset"

classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

plt.figure(figsize=(14, 12))

for i, category in enumerate(classes):

    folder = os.path.join(dataset_path, category)

    images = os.listdir(folder)

    selected_images = random.sample(
        images,
        min(5, len(images))
    )

    for j, image_name in enumerate(selected_images):

        image_path = os.path.join(folder, image_name)

        img = Image.open(image_path).convert("RGB")

        plt.subplot(4, 5, i * 5 + j + 1)

        plt.imshow(img)

        plt.title(category)

        plt.axis("off")

plt.tight_layout()

plt.show()