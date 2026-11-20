from PIL import Image
import pillow_avif

image_path = (
    "wound_dataset/initial/"
    "818fef_2ce0064768e546a68ab1b3665a1b96e6~mv2.avif"
)

try:

    image = Image.open(image_path)

    print("Image opened successfully!")
    print("Format:", image.format)
    print("Dimensions:", image.size)
    print("Mode:", image.mode)

    image.verify()

    print("Image verification successful.")

except Exception as e:

    print("Image verification failed:", e)