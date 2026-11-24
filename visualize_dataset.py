from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import random

# Dataset directory
BASE_DIR = Path("split_dataset/train")

# Classes
CLASSES = [
    "healed",
    "healing",
    "inflammation",
    "initial"
]

# Configuration
SAMPLES_PER_CLASS = 5
IMAGE_SIZE = (220, 220)
PADDING = 15

random.seed(42)

# Contact sheet dimensions
columns = SAMPLES_PER_CLASS
rows = len(CLASSES)

cell_width = IMAGE_SIZE[0]
cell_height = IMAGE_SIZE[1] + 35

sheet_width = columns * cell_width + (columns + 1) * PADDING
sheet_height = rows * cell_height + (rows + 1) * PADDING

# Create blank contact sheet
contact_sheet = Image.new(
    "RGB",
    (sheet_width, sheet_height),
    "white"
)

draw = ImageDraw.Draw(contact_sheet)

for row, class_name in enumerate(CLASSES):

    folder = BASE_DIR / class_name

    image_files = [
        file for file in folder.iterdir()
        if file.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]
    ]

    selected_images = random.sample(
        image_files,
        min(SAMPLES_PER_CLASS, len(image_files))
    )

    for col, image_path in enumerate(selected_images):

        with Image.open(image_path) as img:

            # Convert to RGB
            img = img.convert("RGB")

            # Preserve aspect ratio
            img = ImageOps.contain(img, IMAGE_SIZE)

            # White background
            canvas = Image.new(
                "RGB",
                IMAGE_SIZE,
                "white"
            )

            x_offset = (IMAGE_SIZE[0] - img.width) // 2
            y_offset = (IMAGE_SIZE[1] - img.height) // 2

            canvas.paste(img, (x_offset, y_offset))

        x = PADDING + col * (cell_width + PADDING)
        y = PADDING + row * (cell_height + PADDING)

        contact_sheet.paste(canvas, (x, y))

        # Image filename
        draw.text(
            (x, y + IMAGE_SIZE[1] + 5),
            f"{class_name} - {col + 1}",
            fill="black"
        )

# Save contact sheet
output_path = Path("sample_images_contact_sheet.png")

contact_sheet.save(output_path)

print("Contact sheet created successfully!")
print("Location:", output_path.resolve())