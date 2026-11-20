import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "wound_dataset"

classes = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

data = []

for category in classes:

    folder = os.path.join(dataset_path, category)

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        brightness = np.mean(gray)

        contrast = np.std(gray)

        height, width = gray.shape

        data.append({

            "class": category,

            "filename": filename,

            "brightness": brightness,

            "contrast": contrast,

            "width": width,

            "height": height

        })


df = pd.DataFrame(data)

print(df.head())

print("\nClass-wise statistics:")

print(
    df.groupby("class")[
        ["brightness", "contrast"]
    ].mean()
)

plt.figure(figsize=(10, 6))

for category in classes:

    subset = df[df["class"] == category]

    plt.hist(
        subset["brightness"],
        alpha=0.5,
        label=category,
        bins=20
    )

plt.xlabel("Brightness")

plt.ylabel("Number of Images")

plt.title("Brightness Distribution by Wound Category")

plt.legend()

plt.show()

plt.figure(figsize=(10, 6))

for category in classes:

    subset = df[df["class"] == category]

    plt.hist(
        subset["contrast"],
        alpha=0.5,
        label=category,
        bins=20
    )

plt.xlabel("Contrast")

plt.ylabel("Number of Images")

plt.title("Contrast Distribution by Wound Category")

plt.legend()

plt.show()