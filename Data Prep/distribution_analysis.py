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

data = []

# Calculate brightness and contrast
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

        brightness = gray.mean()

        contrast = gray.std()

        data.append({

            "class": category,
            "filename": filename,
            "brightness": brightness,
            "contrast": contrast

        })

# Create DataFrame
df = pd.DataFrame(data)

print("\nDataset Statistics:\n")

print(
    df.groupby("class")[
        ["brightness", "contrast"]
    ].describe()
)

# Brightness Boxplot
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="class",
    y="brightness"
)

plt.title("Brightness Distribution Across Classes")

plt.xlabel("Wound Category")

plt.ylabel("Brightness")

plt.show()


# Contrast Boxplot
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="class",
    y="contrast"
)

plt.title("Contrast Distribution Across Classes")

plt.xlabel("Wound Category")

plt.ylabel("Contrast")

plt.show()

# Contrast distribution
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="class",
    y="contrast"
)

plt.title("Contrast Distribution Across Classes")
plt.xlabel("Wound Category")
plt.ylabel("Contrast")

plt.tight_layout()
plt.show()