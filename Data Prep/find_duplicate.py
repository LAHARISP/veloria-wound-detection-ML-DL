import os
import csv
from PIL import Image, ImageOps
import imagehash
from itertools import combinations

dataset_path = "wound_dataset"

classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

# Hamming distance threshold
THRESHOLD = 8

# Store image information
image_data = []

print("Starting duplicate detection...\n")

for category in classes:

    folder = os.path.join(dataset_path, category)

    print("Processing:", category)

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as img:

                img = img.convert("RGB")

                normal_hash = imagehash.phash(img)

                # Hash of horizontally flipped image
                mirrored_img = ImageOps.mirror(img)

                mirrored_hash = imagehash.phash(mirrored_img)

                image_data.append({
                    "class": category,
                    "filename": filename,
                    "path": path,
                    "hash": normal_hash,
                    "mirror_hash": mirrored_hash
                })

        except Exception as e:

            print("Could not process:", filename, "|", e)


# Compare images within each class
duplicate_pairs = []

for category in classes:

    class_images = [
        img for img in image_data
        if img["class"] == category
    ]

    print("\nComparing:", category)

    for img1, img2 in combinations(class_images, 2):

        distance_normal = img1["hash"] - img2["hash"]

        distance_mirror = img1["hash"] - img2["mirror_hash"]

        distance = min(distance_normal, distance_mirror)

        if distance <= THRESHOLD:

            duplicate_pairs.append({
                "class": category,
                "image_1": img1["filename"],
                "image_2": img2["filename"],
                "normal_distance": distance_normal,
                "mirror_distance": distance_mirror,
                "best_distance": distance
            })


# Save results
output_file = "duplicate_report.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "class",
            "image_1",
            "image_2",
            "normal_distance",
            "mirror_distance",
            "best_distance"
        ]
    )

    writer.writeheader()
    writer.writerows(duplicate_pairs)


print("\n" + "=" * 60)

print("DUPLICATE DETECTION COMPLETED")

print("Total readable images:", len(image_data))

print("Potential duplicate pairs:", len(duplicate_pairs))

print("Report saved:", output_file)

print("=" * 60)