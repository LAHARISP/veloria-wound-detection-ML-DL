
import os
import shutil
import imagehash

from PIL import Image
from collections import defaultdict

dataset_path = "wound_dataset"

duplicate_folder = "duplicate_images"

os.makedirs(duplicate_folder, exist_ok=True)

hash_groups = defaultdict(list)

# Find duplicates
for category in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, category)

    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        try:

            img = Image.open(path).convert("RGB")

            img_hash = imagehash.phash(img)

            hash_groups[str(img_hash)].append(
                (category, path)
            )

        except Exception:
            continue


# Move duplicate images
for img_hash, images in hash_groups.items():

    if len(images) > 1:

        # Keep the first image
        for category, path in images[1:]:

            destination = os.path.join(
                duplicate_folder,
                category
            )

            os.makedirs(destination, exist_ok=True)

            shutil.move(
                path,
                os.path.join(
                    destination,
                    os.path.basename(path)
                )
            )

            print("Moved:", path)

print("\nDuplicate removal completed!")