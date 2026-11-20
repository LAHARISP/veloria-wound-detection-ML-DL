import os
import pandas as pd

# Read duplicate reports
within = pd.read_csv("duplicate_report.csv")
cross = pd.read_csv("cross_class_duplicates.csv")

# Combine duplicate relationships
pairs = []

for _, row in within.iterrows():

    pairs.append((
        (row["class"], row["image_1"]),
        (row["class"], row["image_2"])
    ))

for _, row in cross.iterrows():

    pairs.append((
        (row["class_1"], row["image_1"]),
        (row["class_2"], row["image_2"])
    ))


# Union-Find implementation
parent = {}

def find(x):

    if x not in parent:
        parent[x] = x

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


# Group duplicate images
for a, b in pairs:
    union(a, b)


# Collect connected groups
groups = {}

for image in parent:

    root = find(image)

    if root not in groups:
        groups[root] = []

    groups[root].append(image)


# Create output records
records = []

for group_number, images in enumerate(groups.values(), start=1):

    group_id = f"G{group_number:03d}"

    labels = sorted(set(image[0] for image in images))

    conflict = len(labels) > 1

    for class_name, filename in images:

        records.append({

            "group_id": group_id,
            "class": class_name,
            "filename": filename,
            "label_conflict": conflict,
            "group_labels": ", ".join(labels)

        })


# Save CSV
df = pd.DataFrame(records)

df.to_csv(
    "duplicate_groups.csv",
    index=False
)


# Summary
print("\n========== DUPLICATE GROUP SUMMARY ==========")

print("Total duplicate groups:", len(groups))

print("Images included in groups:", len(records))

print(
    "Groups with label conflicts:",
    sum(
        len(set(image[0] for image in group)) > 1
        for group in groups.values()
    )
)

print("\nReport saved: duplicate_groups.csv")