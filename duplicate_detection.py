import os
import imagehash
from PIL import Image
from collections import defaultdict

dataset_path = "wound_dataset"

hash_groups = defaultdict(list)

for category in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, category)

    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):

        image_path = os.path.join(folder, filename)

        try:

            img = Image.open(image_path).convert("RGB")

            img_hash = imagehash.phash(img)

            hash_groups[str(img_hash)].append(
                (category, image_path)
            )

        except Exception:

            print("Unable to process:", image_path)

print("\nDuplicate Images:\n")

for img_hash, images in hash_groups.items():

    if len(images) > 1:

        print("\nDuplicate group:")

        for category, path in images:

            print(category, ":", path)