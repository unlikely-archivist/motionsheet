from pathlib import Path

import numpy as np
from PIL import Image

from motionsheet.stack import rgb_stack

OUT = Path(__file__).parent / "output"
H, W, SIZE = 60, 120, 20


def save(img, name):
    OUT.mkdir(exist_ok=True)
    Image.fromarray(img).save(OUT / name)


def test_static_clip_is_neutral_gray():
    rng = np.random.default_rng(0)
    scene = rng.uniform(0.2, 0.8, (H, W))
    img = rgb_stack([scene] * 3)
    save(img, "static.png")
    spread = img.max(axis=2).astype(int) - img.min(axis=2)
    assert spread.max() == 0  # no color fringing at all


def test_moving_square_is_red_then_green_then_blue():
    frames = []
    for i in range(3):
        f = np.zeros((H, W))
        f[20:20 + SIZE, 20 + i * 30:20 + i * 30 + SIZE] = 1.0
        frames.append(f)
    img = rgb_stack(frames)
    save(img, "moving-square.png")
    y = 30
    assert tuple(img[y, 30]) == (255, 0, 0)    # first position: red
    assert tuple(img[y, 60]) == (0, 255, 0)    # second position: green
    assert tuple(img[y, 90]) == (0, 0, 255)    # third position: blue
    assert tuple(img[y, 5]) == (0, 0, 0)       # untouched background stays neutral


def test_rejects_wrong_frame_count():
    import pytest
    with pytest.raises(ValueError):
        rgb_stack([np.zeros((2, 2))] * 4)
