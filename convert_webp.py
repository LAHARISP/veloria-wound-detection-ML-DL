from pathlib import Path
from PIL import Image

BASE_DIR = Path("split_dataset")

converted = 0

for webp_file in BASE_DIR.rglob("*.webp"):

    png_file = webp_file.with_suffix(".png")

    if png_file.exists():
        print(f"Already exists: {png_file}")
        continue

    with Image.open(webp_file) as img:
        img.convert("RGB").save(png_file, "PNG")

    converted += 1
    print(f"Converted: {webp_file.name}")

print("\nConversion completed!")
print("New PNG files:", converted)