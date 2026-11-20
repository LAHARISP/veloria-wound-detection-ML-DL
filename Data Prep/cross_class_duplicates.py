import os
import pandas as pd
import imagehash

from PIL import Image, ImageOps

# AVIF support
try:
    import pillow_avif
except ImportError:
    pass


# Dataset location
dataset_path = "wound_dataset"

# Classes
classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

# Store image information
images = []

print("Scanning dataset...\n")


# Step 1: Read all images and calculate hashes

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    for filename in os.listdir(class_path):

        image_path = os.path.join(class_path, filename)

        try:
            img = Image.open(image_path).convert("RGB")

            # Normal perceptual hash
            normal_hash = imagehash.phash(img)

            # Horizontal mirror perceptual hash
            mirrored_hash = imagehash.phash(
                ImageOps.mirror(img)
            )

            images.append({
                "class": class_name,
                "filename": filename,
                "normal_hash": normal_hash,
                "mirrored_hash": mirrored_hash
            })

        except Exception as e:
            print(f"Skipped: {filename} | {e}")


print("\nTotal readable images:", len(images))


# Step 2: Compare images across different classes

duplicates = []

for i in range(len(images)):

    for j in range(i + 1, len(images)):

        img1 = images[i]
        img2 = images[j]

        # Only compare different classes
        if img1["class"] == img2["class"]:
            continue

        # Direct comparison
        direct_distance = (
            img1["normal_hash"] - img2["normal_hash"]
        )

        # Mirrored comparison
        mirror_distance_1 = (
            img1["normal_hash"] - img2["mirrored_hash"]
        )

        mirror_distance_2 = (
            img1["mirrored_hash"] - img2["normal_hash"]
        )

        best_distance = min(
            direct_distance,
            mirror_distance_1,
            mirror_distance_2
        )

        # Similarity threshold
        if best_distance <= 8:

            duplicates.append({

                "class_1": img1["class"],
                "image_1": img1["filename"],

                "class_2": img2["class"],
                "image_2": img2["filename"],

                "direct_distance": direct_distance,
                "mirror_distance": min(
                    mirror_distance_1,
                    mirror_distance_2
                ),

                "best_distance": best_distance

            })


# Step 3: Save results

df = pd.DataFrame(duplicates)

df.to_csv(
    "cross_class_duplicates.csv",
    index=False
)


print("\nCross-class duplicate pairs:", len(duplicates))

print("\nReport saved as cross_class_duplicates.csv")


if len(duplicates) > 0:

    print("\nPotential cross-class duplicates:\n")

    print(df.to_string(index=False))

else:

    print("\nNo potential cross-class duplicates found.")