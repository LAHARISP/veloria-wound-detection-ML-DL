import os
import imagehash
import matplotlib.pyplot as plt

from PIL import Image
from collections import defaultdict

dataset_path = "wound_dataset"

hash_groups = defaultdict(list)

# Detect duplicates
for category in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, category)

    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        try:

            img = Image.open(path).convert("RGB")

            img_hash = imagehash.phash(img)

            hash_groups[str(img_hash)].append(path)

        except Exception:
            continue


# Visualize duplicate pairs
duplicate_groups = [
    paths for paths in hash_groups.values()
    if len(paths) > 1
]

for index, group in enumerate(duplicate_groups):

    fig, axes = plt.subplots(
        1,
        len(group),
        figsize=(12, 5)
    )

    if len(group) == 1:
        axes = [axes]

    for ax, path in zip(axes, group):

        image = Image.open(path).convert("RGB")

        ax.imshow(image)

        ax.set_title(os.path.basename(path))

        ax.axis("off")

    plt.suptitle(f"Duplicate Group {index + 1}")

    plt.tight_layout()

    plt.show()