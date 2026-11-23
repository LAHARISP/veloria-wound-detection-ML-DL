from pathlib import Path
from PIL import Image
import pandas as pd
from collections import Counter

# =====================================
# 1. PROJECT CONFIGURATION
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "split_dataset"

SPLITS = ["train", "validation", "test"]

CLASSES = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

# =====================================
# 2. INSPECTION
# =====================================

image_records = []

image_extensions = {
    ".jpg", ".jpeg", ".png", ".webp", ".avif"
}

for split in SPLITS:

    print(f"\nInspecting {split.upper()} dataset...")

    for class_name in CLASSES:

        folder = DATASET_DIR / split / class_name

        for image_path in folder.iterdir():

            if image_path.suffix.lower() not in image_extensions:
                continue

            try:
                with Image.open(image_path) as img:

                    image_records.append({
                        "split": split,
                        "class": class_name,
                        "filename": image_path.name,
                        "width": img.width,
                        "height": img.height,
                        "mode": img.mode,
                        "format": img.format
                    })

            except Exception as e:

                print("Invalid image:", image_path)
                print("Error:", e)


# =====================================
# 3. CREATE DATAFRAME
# =====================================

df = pd.DataFrame(image_records)

print("\n====================================")
print("IMAGE INSPECTION SUMMARY")
print("====================================")

print("Total images:", len(df))

# =====================================
# 4. IMAGE DIMENSIONS
# =====================================

print("\nIMAGE DIMENSIONS")

print(
    df.groupby(["width", "height"])
    .size()
    .sort_values(ascending=False)
    .head(15)
)

# =====================================
# 5. IMAGE COLOUR MODES
# =====================================

print("\nIMAGE COLOUR MODES")

print(df["mode"].value_counts())

# =====================================
# 6. IMAGE FORMATS
# =====================================

print("\nIMAGE FORMATS")

print(df["format"].value_counts())

# =====================================
# 7. SPLIT DISTRIBUTION
# =====================================

print("\nSPLIT DISTRIBUTION")

print(df.groupby(["split", "class"]).size())

# =====================================
# 8. SAVE REPORT
# =====================================

output_path = PROJECT_DIR / "image_inspection_report.csv"

df.to_csv(output_path, index=False)

print("\nInspection report saved:")
print(output_path)

print("\nImage inspection completed successfully!")