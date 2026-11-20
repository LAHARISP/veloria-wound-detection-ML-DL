import os
from PIL import Image
import imagehash
import matplotlib.pyplot as plt

# Dataset location
dataset_path = "wound_dataset"

# Potential duplicate pairs identified earlier
duplicate_pairs = {

    "healing": [
        ("cut (16).jpg", "cut (41).jpg"),
        ("cut (19).jpg", "cut (34).jpg"),
        ("cut (22).jpg", "cut (4).jpg"),
        ("download.jpg", "Scab.jpg"),
        ("mirrored_cut (19).jpg", "mirrored_cut (34).jpg")
    ],

    "inflammation": [
        ("mirrored_bruises (122).jpg", "mirrored_bruises (98).jpg"),
        ("mirrored_bruises (27).jpg", "mirrored_bruises (83).jpg"),
        ("mirrored_bruises (29).jpg", "mirrored_bruises (86).jpg"),
        ("mirrored_bruises (53).jpg", "mirrored_bruises (69).jpg"),
        ("mirrored_bruises (67).jpg", "mirrored_bruises (73).jpg")
    ]
}

# Store verification results
results = []

for category, pairs in duplicate_pairs.items():

    print("\n" + "=" * 60)
    print("CLASS:", category)
    print("=" * 60)

    folder = os.path.join(dataset_path, category)

    for file1, file2 in pairs:

        path1 = os.path.join(folder, file1)
        path2 = os.path.join(folder, file2)

        # Check whether both files exist
        if not os.path.exists(path1) or not os.path.exists(path2):

            print("\nMissing image:", file1, file2)
            continue

        # Load images
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        # Calculate perceptual hashes
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)

        # Calculate Hamming distance
        distance = hash1 - hash2

        # Print information
        print("\nImage 1:", file1)
        print("Image 2:", file2)

        print("Image 1 dimensions:", img1.size)
        print("Image 2 dimensions:", img2.size)

        print("pHash 1:", hash1)
        print("pHash 2:", hash2)

        print("Hamming distance:", distance)

        results.append({
            "class": category,
            "image_1": file1,
            "image_2": file2,
            "hamming_distance": distance
        })

        # Display images side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img1)
        axes[0].set_title(file1)
        axes[0].axis("off")

        axes[1].imshow(img2)
        axes[1].set_title(file2)
        axes[1].axis("off")

        fig.suptitle(
            f"{category.upper()} | Hamming Distance: {distance}"
        )

        plt.tight_layout()
        plt.show()

# Final summary
print("\n\nDUPLICATE VERIFICATION SUMMARY")
print("=" * 60)

for result in results:

    print(
        result["class"],
        "|",
        result["image_1"],
        "|",
        result["image_2"],
        "| Distance:",
        result["hamming_distance"]
    )
    