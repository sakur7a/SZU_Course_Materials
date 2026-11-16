"""Regenerate the two authored raster signs on Windows; the font is not copied.

Requires Pillow and local Windows fonts. The generated PNGs are checked-in inputs
to build.py, so running Blender itself does not require Pillow or system fonts.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

school = Image.new("RGB", (256, 1024), (211, 212, 203))
draw = ImageDraw.Draw(school)
font = ImageFont.truetype("C:/Windows/Fonts/STXINGKA.TTF", 208)
for i, char in enumerate("深圳大学"):
    draw.text((128, 128 + i * 248), char, font=font, fill=(36, 39, 36), anchor="mm")
school.save(HERE / "school_name.png", optimize=True)

address = Image.new("RGB", (512, 256), (27, 84, 137))
draw = ImageDraw.Draw(address)
draw.rectangle((12, 12, 499, 243), outline=(211, 227, 232), width=6)
font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 145)
draw.text((256, 130), "3688", font=font, fill=(234, 243, 246), anchor="mm")
address.save(HERE / "address.png", optimize=True)
