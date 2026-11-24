from pathlib import Path
from PIL import Image, ImageDraw
import random

BASE_DIR = Path("preprocessing_preview")

CLASSES = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

SAMPLES = 5
SIZE = (224, 224)
PADDING = 15

random.seed(42)

columns = SAMPLES
rows = len(CLASSES)

cell_width = 224
cell_height = 250

sheet = Image.new(
    "RGB",
    (
        columns * cell_width + (columns + 1) * PADDING,
        rows * cell_height + (rows + 1) * PADDING
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)

for row, class_name in enumerate(CLASSES):

    folder = BASE_DIR / class_name

    images = list(folder.glob("*.png"))

    selected = random.sample(
        images,
        min(SAMPLES, len(images))
    )

    for col, image_path in enumerate(selected):

        with Image.open(image_path) as img:
            img = img.convert("RGB")

            x = PADDING + col * (cell_width + PADDING)
            y = PADDING + row * (cell_height + PADDING)

            sheet.paste(img, (x, y))

            draw.text(
                (x, y + 228),
                f"{class_name} - {col + 1}",
                fill="black"
            )

output = "resized_contact_sheet.png"

sheet.save(output)

print("Contact sheet generated:", output)