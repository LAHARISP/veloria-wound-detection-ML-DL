import os
import pandas as pd

dataset_path = "wound_dataset"

classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

# Original image counts
print("\n========== ORIGINAL DATASET ==========")

total_images = 0

for class_name in classes:

    folder = os.path.join(dataset_path, class_name)

    count = len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ])

    total_images += count

    print(f"{class_name}: {count}")

print("Total images:", total_images)


# Within-class duplicate pairs
within = pd.read_csv("duplicate_report.csv")

print("\n========== WITHIN-CLASS DUPLICATES ==========")

print(within.groupby("class").size())

print("Total duplicate pairs:", len(within))


# Cross-class duplicate pairs
cross = pd.read_csv("cross_class_duplicates.csv")

print("\n========== CROSS-CLASS DUPLICATES ==========")

print(cross)

print("Total cross-class pairs:", len(cross))