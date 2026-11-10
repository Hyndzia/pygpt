"""Generuje ikony-placeholdery dla wszystkich głosów do katalogu assets/."""

import hashlib
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import voices

SIZE = 128
FONT_PATH = r"C:\Windows\Fonts\segoeuib.ttf"

# Ciemne, czerwonawe tła — w rodzinie motywu, ale każda postać ma inny odcień.
BASES = [
    (38, 14, 16), (44, 16, 18), (34, 12, 14), (48, 18, 20),
    (40, 13, 15), (30, 13, 17), (46, 15, 15), (36, 16, 19),
]


def initials(label: str) -> str:
    words = [w for w in label.replace("→", " ").replace("-", " ").split() if w]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:1].upper()
    return (words[0][:1] + words[1][:1]).upper()


def base_color(voice_id: str):
    digest = hashlib.md5(voice_id.encode("utf-8")).hexdigest()
    return BASES[int(digest[:2], 16) % len(BASES)]


def make_icon(voice_id: str, text: str):
    bg = base_color(voice_id)
    ring = tuple(min(255, c + 62) for c in bg)
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    pad = 3
    draw.ellipse((pad, pad, SIZE - pad, SIZE - pad), fill=bg + (255,),
                 outline=ring + (255,), width=3)

    font_size = 54 if len(text) == 1 else 44
    font = ImageFont.truetype(FONT_PATH, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (SIZE - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (SIZE - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(238, 238, 240, 255))
    return image


if __name__ == "__main__":
    out_dir = voices.ASSETS_DIR
    os.makedirs(out_dir, exist_ok=True)
    force = "--force" in sys.argv
    made = skipped = 0
    for voice_id in voices.voice_ids():
        path = os.path.join(out_dir, voice_id + ".png")
        if os.path.exists(path) and not force:
            skipped += 1          # nie nadpisujemy podmienionych ikon
            continue
        make_icon(voice_id, initials(voices.label(voice_id))).save(path)
        made += 1
    print(f"wygenerowano {made}, pominieto {skipped} (uzyj --force, aby nadpisac)")
