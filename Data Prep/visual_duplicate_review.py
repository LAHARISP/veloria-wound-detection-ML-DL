import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "wound_dataset"

# Read duplicate report
df = pd.read_csv("duplicate_report.csv")

# Display 8 pairs at a time
batch_size = 8

for start in range(0, len(df), batch_size):

    batch = df.iloc[start:start + batch_size]

    fig, axes = plt.subplots(
        len(batch), 2,
        figsize=(10, 4 * len(batch))
    )

    if len(batch) == 1:
        axes = [axes]

    for i, (_, row) in enumerate(batch.iterrows()):

        folder = os.path.join(
            dataset_path,
            row["class"]
        )

        path1 = os.path.join(folder, row["image_1"])
        path2 = os.path.join(folder, row["image_2"])

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        axes[i][0].imshow(img1)
        axes[i][0].set_title(row["image_1"])
        axes[i][0].axis("off")

        axes[i][1].imshow(img2)
        axes[i][1].set_title(row["image_2"])
        axes[i][1].axis("off")

        print(
            row["class"],
            "|",
            row["image_1"],
            "|",
            row["image_2"],
            "| Distance:",
            row["best_distance"]
        )

    plt.tight_layout()
    plt.show()