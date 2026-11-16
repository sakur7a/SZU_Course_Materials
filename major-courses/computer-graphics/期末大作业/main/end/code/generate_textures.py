"""生成项目使用的原创、可平铺 PPM 纹理；运行程序不依赖 Python。

重建这些纹理时需要 Pillow 和 Windows 自带的微软雅黑字体。随机种子固定。
"""
from pathlib import Path
from random import Random
from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path(__file__).resolve().parent / "assets"
OUTPUT.mkdir(exist_ok=True)


def clamp(value):
    return max(0, min(255, int(value)))


def material(name, base, noise, seam=None, seed=1):
    rng = Random(seed)
    image = Image.new("RGB", (256, 256))
    pixels = image.load()
    for y in range(256):
        for x in range(256):
            value = rng.gauss(0, noise)
            if seam and (x % seam == 0 or y % seam == 0):
                value -= 18
            pixels[x, y] = tuple(clamp(v + value) for v in base)
    image.save(OUTPUT / (name + ".ppm"))


material("concrete", (224, 225, 218), 4.0, None, 11)
material("stone", (192, 188, 178), 4.8, 64, 12)
material("paving", (219, 215, 203), 3.5, 64, 13)
# grass.ppm is derived separately from ambientCG Grass004 (CC0); see assets/CREDITS.txt.
material("glass", (112, 143, 153), 2.2, None, 15)
material("water", (91, 137, 145), 4.5, None, 16)
material("white", (255, 255, 255), 0.0, None, 17)


def nameplate(filename, chinese, english):
    image = Image.new("RGB", (512, 160), (226, 227, 220))
    draw = ImageDraw.Draw(image)
    chinese_font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 61)
    english_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    for value, font, y in [(chinese, chinese_font, 17), (english, english_font, 100)]:
        bounds = draw.textbbox((0, 0), value, font=font)
        x = (512 - (bounds[2] - bounds[0])) // 2
        draw.text((x, y), value, font=font, fill=(31, 51, 58))
    image.save(OUTPUT / filename)


nameplate("huidian.ppm", "汇典楼", "HUIDIAN LIBRARY")
nameplate("huizhi.ppm", "汇智楼", "HUIZHI LIBRARY")
print("Textures written to", OUTPUT)
