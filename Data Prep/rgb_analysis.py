import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = "wound_dataset"

classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

rgb_data = []

for category in classes:

    folder = os.path.join(dataset_path, category)

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        red_mean = image[:, :, 0].mean()
        green_mean = image[:, :, 1].mean()
        blue_mean = image[:, :, 2].mean()

        rgb_data.append({
            "class": category,
            "filename": filename,
            "red": red_mean,
            "green": green_mean,
            "blue": blue_mean
        })

rgb_df = pd.DataFrame(rgb_data)

print("\nRGB Channel Statistics:")

print(
    rgb_df.groupby("class")[
        ["red", "green", "blue"]
    ].mean()
)

print("\nTotal images analysed:", len(rgb_df))

channels = ["red", "green", "blue"]

for channel in channels:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=rgb_df,
        x="class",
        y=channel
    )

    plt.title(
        f"{channel.capitalize()} Channel Distribution Across Classes"
    )

    plt.xlabel("Wound Category")
    plt.ylabel("Pixel Intensity (0-255)")

    plt.tight_layout()
    plt.show()