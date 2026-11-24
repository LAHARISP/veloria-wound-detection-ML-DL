from pathlib import Path
from PIL import Image, ImageOps

BASE_DIR = Path("split_dataset/train")

OUTPUT_DIR = Path("preprocessing_preview")

CLASSES = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

TARGET_SIZE = (224, 224)

OUTPUT_DIR.mkdir(exist_ok=True)

for class_name in CLASSES:

    input_folder = BASE_DIR / class_name
    output_folder = OUTPUT_DIR / class_name

    output_folder.mkdir(exist_ok=True)

    images = list(input_folder.iterdir())

    for image_path in images:

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]:
            continue

        try:

            with Image.open(image_path) as img:

                img = img.convert("RGB")

                # Preserve aspect ratio
                img = ImageOps.contain(
                    img,
                    TARGET_SIZE
                )

                # Add padding
                canvas = Image.new(
                    "RGB",
                    TARGET_SIZE,
                    (0, 0, 0)
                )

                x = (224 - img.width) // 2
                y = (224 - img.height) // 2

                canvas.paste(img, (x, y))

                output_path = (
                    output_folder /
                    f"{image_path.stem}.png"
                )

                canvas.save(output_path)

        except Exception as e:

            print("Error:", image_path, e)

print("Resizing preview completed!")