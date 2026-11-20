from PIL import Image
import pillow_avif
import os

source = (
    "wound_dataset/initial/"
    "818fef_2ce0064768e546a68ab1b3665a1b96e6~mv2.avif"
)

destination = (
    "wound_dataset/initial/"
    "818fef_2ce0064768e546a68ab1b3665a1b96e6_converted.png"
)

image = Image.open(source)

image.convert("RGB").save(destination, "PNG")

print("Converted successfully!")
print("Saved at:", destination)