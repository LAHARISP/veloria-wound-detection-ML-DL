import os
from PIL import Image
from collections import Counter

dataset_path = "wound_dataset"

classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

total_images = 0
image_dimensions = Counter()
image_formats = Counter()

for category in classes:

    folder = os.path.join(dataset_path, category)

    count = 0

    print(f"\nProcessing: {category}")

    for filename in os.listdir(folder):

        image_path = os.path.join(folder, filename)

        try:

            with Image.open(image_path) as img:

                img.verify()

            with Image.open(image_path) as img:

                image_dimensions[img.size] += 1

                image_formats[img.format] += 1

                count += 1

        except Exception:

            print("Corrupted image:", image_path)

    print("Images:", count)

    total_images += count

print("\n----------------------------")

print("Total images:", total_images)

print("\nImage dimensions:")

for dimension, count in image_dimensions.items():

    print(dimension, ":", count)

print("\nImage formats:")

for format_name, count in image_formats.items():

    print(format_name, ":", count)