import os
import cv2
import matplotlib.pyplot as plt

dataset_path = "wound_dataset/healing"

brightness_values = []

for filename in os.listdir(dataset_path):

    path = os.path.join(dataset_path, filename)

    image = cv2.imread(path)

    if image is None:
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = gray.mean()

    brightness_values.append(
        (filename, brightness)
    )

# Sort by brightness
brightness_values.sort(key=lambda x: x[1])

print("10 Darkest Images:")

for filename, brightness in brightness_values[:10]:

    print(filename, ":", brightness)