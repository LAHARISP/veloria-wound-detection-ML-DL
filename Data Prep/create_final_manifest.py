import os
import pandas as pd

dataset_path = "wound_dataset"

classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

# Read duplicate reports
within = pd.read_csv("duplicate_report.csv")
cross = pd.read_csv("cross_class_duplicates.csv")

parent = {}

# Register every image
all_images = []

for class_name in classes:

    folder = os.path.join(dataset_path, class_name)

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        if os.path.isfile(path):

            image = (class_name, filename)

            all_images.append(image)
            parent[image] = image


# Union-Find
def find(x):

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


# Within-class duplicate relationships
for _, row in within.iterrows():

    a = (row["class"], row["image_1"])
    b = (row["class"], row["image_2"])

    union(a, b)


# Cross-class duplicate relationships
for _, row in cross.iterrows():

    a = (row["class_1"], row["image_1"])
    b = (row["class_2"], row["image_2"])

    union(a, b)


# Explicitly group original AVIF and converted PNG
avif_pair = (
    "initial",
    "818fef_2ce0064768e546a68ab1b3665a1b96e6~mv2.avif"
)

png_pair = (
    "initial",
    "818fef_2ce0064768e546a68ab1b3665a1b96e6_converted.png"
)

if avif_pair in parent and png_pair in parent:
    union(avif_pair, png_pair)


# Build groups, including singleton images
groups = {}

for image in all_images:

    root = find(image)

    groups.setdefault(root, []).append(image)


# Create manifest
records = []

for number, group in enumerate(groups.values(), start=1):

    group_id = f"G{number:03d}"

    labels = sorted(set(image[0] for image in group))

    conflict = len(labels) > 1

    for class_name, filename in group:

        records.append({

            "group_id": group_id,
            "class": class_name,
            "filename": filename,
            "label_conflict": conflict,
            "group_labels": ", ".join(labels),
            "group_size": len(group)

        })


df = pd.DataFrame(records)

df.to_csv(
    "final_dataset_manifest.csv",
    index=False
)


# Summary
print("\n========== FINAL DATASET MANIFEST ==========")

print("Total files:", len(df))

print("Total groups:", df["group_id"].nunique())

print(
    "Singleton groups:",
    (df.groupby("group_id").size() == 1).sum()
)

print(
    "Duplicate groups:",
    (df.groupby("group_id").size() > 1).sum()
)

print(
    "Label conflict groups:",
    df[df["label_conflict"]]["group_id"].nunique()
)

print("\nClass-wise file counts:")

print(df["class"].value_counts())

print("\nManifest saved: final_dataset_manifest.csv")