import os
import cv2
import matplotlib.pyplot as plt

dataset_path = "wound_dataset/healing"

darkest_images = [
    "cut (23).jpg",
    "mirrored_cut (23).jpg",
    "medetec_0065.png",
    "wsnet_0370.png",
    "wsnet_0368.png"
]

plt.figure(figsize=(15, 5))

for i, filename in enumerate(darkest_images):

    path = os.path.join(dataset_path, filename)

    image = cv2.imread(path)

    if image is None:
        print("Could not load:", filename)
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.title(filename, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()