import pandas as pd
import shutil

from pathlib import Path
from sklearn.model_selection import train_test_split


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = PROJECT_DIR / "Data Prep" / "model_dataset_manifest.csv"

OUTPUT_DIR = PROJECT_DIR / "split_dataset"

OUTPUT_MANIFEST = PROJECT_DIR / "split_dataset_manifest.csv"


# ==========================================
# 2. CONFIGURATION
# ==========================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]


# ==========================================
# 3. LOAD DATASET MANIFEST
# ==========================================

print("\nLoading model dataset manifest...")

df = pd.read_csv(MANIFEST_PATH)

print("Manifest loaded successfully.")

print("\nTotal images:", len(df))


# ==========================================
# 4. VALIDATE DATASET
# ==========================================

required_columns = [
    "group_id",
    "class",
    "output_path"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")


if df["group_id"].duplicated().any():
    raise ValueError("Duplicate group IDs found!")


if set(df["class"].unique()) != set(CLASSES):
    raise ValueError("Unexpected class names found!")


if not df["output_path"].apply(
    lambda path: Path(path).is_file()
).all():
    raise FileNotFoundError(
        "One or more source images are missing."
    )


print("\nDataset verification passed.")


# ==========================================
# 5. CREATE TRAIN / TEMPORARY SPLIT
# ==========================================

print("\nCreating training and temporary datasets...")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["class"]
)


# ==========================================
# 6. CREATE VALIDATION / TEST SPLIT
# ==========================================

print("\nCreating validation and test datasets...")

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["class"]
)


# ==========================================
# 7. ASSIGN SPLIT LABELS
# ==========================================

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


# Combine all records

final_df = pd.concat(
    [train_df, validation_df, test_df],
    ignore_index=True
)


# ==========================================
# 8. CHECK GROUP SEPARATION
# ==========================================

train_groups = set(train_df["group_id"])
validation_groups = set(validation_df["group_id"])
test_groups = set(test_df["group_id"])

assert train_groups.isdisjoint(validation_groups)
assert train_groups.isdisjoint(test_groups)
assert validation_groups.isdisjoint(test_groups)

assert len(final_df) == len(df)

print("\nNo group leakage detected.")


# ==========================================
# 9. CREATE OUTPUT DIRECTORIES
# ==========================================

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. "
        "Rename or move the existing folder before running again."
    )

print("\nCreating split directories...")

for split in ["train", "validation", "test"]:

    for class_name in CLASSES:

        folder = OUTPUT_DIR / split / class_name

        folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 10. COPY IMAGES
# ==========================================

print("\nCopying images...")

for _, row in final_df.iterrows():

    source = Path(row["output_path"])

    destination = (
        OUTPUT_DIR
        / row["split"]
        / row["class"]
        / row["output_filename"]
    )

    shutil.copy2(source, destination)

    final_df.loc[
        final_df["group_id"] == row["group_id"],
        "split_output_path"
    ] = str(destination)


# ==========================================
# 11. SAVE SPLIT MANIFEST
# ==========================================

final_df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)

print("\nSplit manifest generated:")
print(OUTPUT_MANIFEST)


# ==========================================
# 12. FINAL DATASET SUMMARY
# ==========================================

print("\n===================================")
print("FINAL DATASET SPLIT SUMMARY")
print("===================================")

for split in ["train", "validation", "test"]:

    subset = final_df[final_df["split"] == split]

    print(f"\n{split.upper()} DATASET")
    print("---------------------------")

    print(
        subset["class"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
    )

    print("Total:", len(subset))


print("\n===================================")
print("TOTAL IMAGES:", len(final_df))
print("UNIQUE GROUPS:", final_df["group_id"].nunique())
print("RANDOM STATE:", RANDOM_STATE)
print("===================================")

print("\nDataset splitting completed successfully!")