"""Generate a synthetic pasta-shape image dataset.

Draws procedural shapes standing in for spaghetti / tagliatelle / fusilli / penne.
No real dataset needed, fully reproducible, and small enough to track with DVC.
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["spaghetti", "tagliatelle", "fusilli", "penne"]
IMG_SIZE = 64
IMAGES_PER_CLASS = 200
OUT_DIR = "data/pasta"
SEED = 42


def draw_spaghetti(d, w, h):
    y = h // 2 + random.randint(-8, 8)
    jitter = random.randint(-3, 3)
    d.line([(0, y), (w, y + jitter)], fill=255, width=3)


def draw_tagliatelle(d, w, h):
    phase = random.uniform(0, 2 * np.pi)
    pts = [(x, h // 2 + 15 * np.sin(x / 8 + phase)) for x in range(0, w, 2)]
    d.line(pts, fill=255, width=8)


def draw_fusilli(d, w, h):
    cx = w // 2 + random.randint(-5, 5)
    pts = [(cx + 18 * np.cos(t / 4), h // 2 + t) for t in range(-h // 2 + 2, h // 2 - 2, 2)]
    d.line(pts, fill=255, width=4)


def draw_penne(d, w, h):
    x0 = random.randint(8, w - 40)
    y0 = h // 2 + random.randint(-10, 10)
    d.line([(x0, y0 - 15), (x0 + 30, y0 + 15)], fill=255, width=10)


DRAW = {
    "spaghetti": draw_spaghetti,
    "tagliatelle": draw_tagliatelle,
    "fusilli": draw_fusilli,
    "penne": draw_penne,
}


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    for cls in CLASSES:
        cls_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
            d = ImageDraw.Draw(img)
            DRAW[cls](d, IMG_SIZE, IMG_SIZE)
            img.save(os.path.join(cls_dir, f"{i}.png"))
        print(f"generated {IMAGES_PER_CLASS} images for '{cls}'")

    print(f"done. dataset at {OUT_DIR}/<class>/*.png")


if __name__ == "__main__":
    main()
