import os

dataset_path = "wound_dataset"

classes = ["healing", "inflammation"]

for category in classes:

    folder = os.path.join(dataset_path, category)

    print("\n" + "=" * 60)
    print("FOLDER:", folder)
    print("=" * 60)

    if not os.path.exists(folder):
        print("Folder does not exist!")
        continue

    files = os.listdir(folder)

    print("Total files:", len(files))

    print("\nFiles containing 'cut' or 'bruise':")

    for filename in files:

        if "cut" in filename.lower() or "bruise" in filename.lower():
            print(filename)

    print("\nFirst 15 filenames:")

    for filename in files[:15]:
        print(filename)