import os
import cv2

dataset_path = "wound_dataset"

classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

total_images = 0
total_readable = 0
total_skipped = 0

for category in classes:

    folder = os.path.join(dataset_path, category)

    readable = 0
    skipped = 0

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        total_images += 1

        image = cv2.imread(path)

        if image is None:

            print("Cannot read:", category, filename)
            skipped += 1

        else:
            readable += 1

    total_readable += readable
    total_skipped += skipped

    print(f"\n{category}")
    print("Readable:", readable)
    print("Skipped:", skipped)

print("\nTOTAL SUMMARY")
print("Total files:", total_images)
print("Readable:", total_readable)
print("Skipped:", total_skipped)