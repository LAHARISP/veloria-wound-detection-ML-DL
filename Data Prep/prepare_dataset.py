import pandas as pd
import shutil
from pathlib import Path
from PIL import Image

# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "wound_dataset"

MANIFEST_PATH = PROJECT_DIR / "final_dataset_manifest.csv"

OUTPUT_DIR = PROJECT_DIR / "model_dataset"

OUTPUT_MANIFEST = PROJECT_DIR / "model_dataset_manifest.csv"


# =====================================================
# 2. IMAGE FORMAT PRIORITY
# =====================================================

# Lower number = higher preference
# PNG is preferred over AVIF because it is more portable.

FORMAT_PRIORITY = {
    ".jpg": 1,
    ".jpeg": 1,
    ".png": 2,
    ".webp": 3,
    ".avif": 4
}


# =====================================================
# 3. LOAD ORIGINAL MANIFEST
# =====================================================

print("\nLoading original dataset manifest...")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

df = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "group_id",
    "class",
    "filename",
    "label_conflict",
    "group_labels",
    "group_size"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in manifest: {missing_columns}"
    )

print("Manifest loaded successfully.")

print("\nOriginal image count:", len(df))


# =====================================================
# 4. IDENTIFY LABEL CONFLICT GROUPS
# =====================================================

def is_conflict(value):
    """
    Convert manifest conflict values into Boolean values.
    """

    return str(value).strip().lower() in [
        "true", "1", "yes"
    ]


df["label_conflict"] = df["label_conflict"].apply(is_conflict)

conflict_groups = df.loc[
    df["label_conflict"], "group_id"
].unique()

print("\nLabel conflict groups:")
print(conflict_groups)

print("Number of conflict groups:", len(conflict_groups))

conflict_images = df[
    df["group_id"].isin(conflict_groups)
]

print("Images excluded due to label conflicts:",
      len(conflict_images))


# =====================================================
# 5. REMOVE CONFLICT GROUPS FROM MODEL DATA
# =====================================================

clean_df = df[
    ~df["group_id"].isin(conflict_groups)
].copy()

print("\nImages remaining after conflict exclusion:",
      len(clean_df))


# =====================================================
# 6. CHECK GROUP LABEL CONSISTENCY
# =====================================================

# A group should contain images belonging to only one class.
# This is an additional safety check.

group_class_counts = clean_df.groupby(
    "group_id"
)["class"].nunique()

invalid_groups = group_class_counts[
    group_class_counts > 1
].index

if len(invalid_groups) > 0:
    raise ValueError(
        "Unexpected cross-class groups remain: "
        f"{list(invalid_groups)}"
    )


# =====================================================
# 7. SELECT ONE IMAGE PER GROUP
# =====================================================

def get_format_priority(filename):
    extension = Path(filename).suffix.lower()

    return FORMAT_PRIORITY.get(extension, 99)


selected_rows = []

for group_id, group in clean_df.groupby("group_id"):

    group = group.copy()

    # Prefer portable image formats
    group["format_priority"] = group[
        "filename"
    ].apply(get_format_priority)

    # Select the best available representative
    group = group.sort_values(
        by=["format_priority", "filename"]
    )

    selected_rows.append(group.iloc[0])


selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

print("\nUnique images selected:", len(selected_df))


# =====================================================
# 8. CREATE OUTPUT DIRECTORIES
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

classes = sorted(selected_df["class"].unique())

for class_name in classes:
    (OUTPUT_DIR / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

print("\nOutput directories created.")


# =====================================================
# 9. COPY SELECTED IMAGES
# =====================================================

output_records = []

print("\nCopying selected images...\n")

for _, row in selected_df.iterrows():

    class_name = str(row["class"])
    filename = str(row["filename"])
    group_id = str(row["group_id"])

    source_path = DATASET_DIR / class_name / filename

    if not source_path.exists():
        raise FileNotFoundError(
            f"Image not found: {source_path}"
        )

    # Verify that the selected image can be opened.
    try:
        with Image.open(source_path) as img:
            img.verify()

        with Image.open(source_path) as img:
            width, height = img.size
            image_format = img.format

    except Exception as error:
        raise ValueError(
            f"Invalid image: {source_path}\n{error}"
        )

    # Prefix with group ID to avoid filename collisions.
    output_filename = (
        f"{group_id}__{source_path.name}"
    )

    destination_path = (
        OUTPUT_DIR / class_name / output_filename
    )

    shutil.copy2(
        source_path,
        destination_path
    )

    output_records.append({
        "group_id": group_id,
        "class": class_name,
        "original_filename": filename,
        "output_filename": output_filename,
        "source_path": str(source_path),
        "output_path": str(destination_path),
        "width": width,
        "height": height,
        "format": image_format
    })


# =====================================================
# 10. GENERATE NEW MANIFEST
# =====================================================

output_df = pd.DataFrame(output_records)

output_df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)

print("\nNew manifest generated:")
print(OUTPUT_MANIFEST)


# =====================================================
# 11. CLASS-WISE IMAGE COUNT
# =====================================================

print("\n===================================")
print("FINAL DATASET SUMMARY")
print("===================================")

class_counts = output_df["class"].value_counts().sort_index()

for class_name, count in class_counts.items():

    print(f"{class_name:<20}: {count}")

print("-----------------------------------")

print("Total unique images:", len(output_df))

print("Excluded conflict groups:", len(conflict_groups))

print("Excluded conflict images:", len(conflict_images))

print("Number of classes:", output_df["class"].nunique())

print("===================================")


# =====================================================
# 12. FINAL VALIDATION
# =====================================================

# Check that every selected group appears only once.
assert output_df["group_id"].is_unique

# Check that no conflict group was selected.
assert not set(output_df["group_id"]).intersection(
    set(conflict_groups)
)

# Check that copied images exist.
assert all(
    Path(path).exists()
    for path in output_df["output_path"]
)

print("\nDataset preparation completed successfully!")

print("\nOriginal dataset remains unchanged.")
print("Clean dataset location:", OUTPUT_DIR)