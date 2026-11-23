from pathlib import Path

BASE_DIR = Path("split_dataset")

for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()}")

    for class_folder in (BASE_DIR / split).iterdir():

        if not class_folder.is_dir():
            continue

        for file in class_folder.iterdir():

            if file.suffix.lower() not in [
                ".jpg", ".jpeg", ".png"
            ]:
                print(
                    f"Check this file: {file}"
                )