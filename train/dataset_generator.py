#!/usr/bin/env python3
#

from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor


class DatasetGenerator:
    def __init__(self, background_dir, foreground_dir):
        self._background_dir = Path(background_dir)
        self._foreground_dir = Path(foreground_dir)
        self._background_crop_top_left = (100, 50)
        self._background_crop_size = (1700, 800)

        self._background_images = []
        self._foreground_images = []

    def load_image(self, d):
        image = Image.open(d)
        image = ToTensor()(image)
        return image

    def load_background_image(self, d):
        image = self.load_image(d)
        left, top = self._background_crop_top_left
        bottom = top + self._background_crop_size[1]
        right = left + self._background_crop_size[0]
        image = image[
            :,
            top:bottom,
            left:right,
        ]
        return image

    def load_images(self):
        for f in self._background_dir.iterdir():
            background_image = self.load_background_image(f)
            self._background_images.append(background_image)
        for f in self._foreground_dir.iterdir():
            self._foreground_images.append(self.load_image(f))

    def debug_dump(self):
        output = Path("/tmp/debug_dump")
        output.mkdir(exist_ok=True)
        for i, img in enumerate(self._background_images):
            torchvision.utils.save_image(img, output / f"background_{i}.png")
        for i, img in enumerate(self._foreground_images):
            torchvision.utils.save_image(img, output / f"foreground_{i}.png")


if __name__ == "__main__":
    background_dir = "../../datasets/background/cave/"
    foreground_dir = "../../datasets/foreground/cave/"
    d = DatasetGenerator(background_dir, foreground_dir=foreground_dir)
    d.load_images()
    d.debug_dump()
