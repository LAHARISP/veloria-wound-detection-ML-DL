from pathlib import Path

BASE_DIR = Path("split_dataset")

splits = ["train", "validation", "test"]
classes = ["healed", "healing", "inflammation", "initial"]

print("\nSPLIT DATASET VERIFICATION")
print("=" * 45)

grand_total = 0

for split in splits:
    split_total = 0

    print(f"\n{split.upper()}")

    for class_name in classes:
        folder = BASE_DIR / split / class_name

        if not folder.exists():
            print(f"{class_name}: FOLDER MISSING")
            continue

        images = [
            file for file in folder.iterdir()
            if file.is_file() and file.suffix.lower()
            in [".jpg", ".jpeg", ".png", ".webp", ".avif"]
        ]

        count = len(images)
        split_total += count

        print(f"{class_name}: {count}")

    print(f"Total {split}: {split_total}")

    grand_total += split_total

print("\n" + "=" * 45)
print("GRAND TOTAL:", grand_total)