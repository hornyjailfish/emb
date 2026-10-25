"""Unit tests for VPR geometry transforms (no model, no DB)."""

import unittest

from PIL import Image

from src.preprocess import (
    TransformMode,
    center_crop,
    letterbox,
    squash,
    transform_image,
)


def _img(w: int, h: int, color=(200, 50, 50)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


class TransformTests(unittest.TestCase):
    def test_squash_exact_size_ignores_aspect(self):
        out = squash(_img(640, 320), 322)
        self.assertEqual(out.size, (322, 322))

    def test_letterbox_pads_short_axis(self):
        # Wide image → letterbox should pad top/bottom; center row is content.
        src = _img(640, 160, color=(255, 0, 0))
        out = letterbox(src, 320, pad_color=(0, 0, 0))
        self.assertEqual(out.size, (320, 320))
        # Corners should be pad (black); middle should be red-ish content.
        self.assertEqual(out.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(out.getpixel((160, 160))[0], 255)

    def test_center_crop_square(self):
        out = center_crop(_img(800, 400), 200)
        self.assertEqual(out.size, (200, 200))

    def test_center_crop_upscales_small(self):
        out = center_crop(_img(50, 40), 100)
        self.assertEqual(out.size, (100, 100))

    def test_transform_image_modes(self):
        src = _img(400, 200)
        for mode in TransformMode:
            with self.subTest(mode=mode.value):
                out = transform_image(src, 128, mode)
                self.assertEqual(out.size, (128, 128))
                self.assertEqual(out.mode, "RGB")

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            transform_image(_img(10, 10), 8, "nope")


if __name__ == "__main__":
    unittest.main()
