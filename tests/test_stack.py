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


def test_true_color_keeps_still_areas_in_video_color():
    from motionsheet.stack import true_color
    rng = np.random.default_rng(1)
    color = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    gray = color.mean(axis=2).astype(np.uint8)
    stack = rgb_stack([gray / 255] * 3)
    assert (true_color(stack, color, [gray] * 3) == color).all()


def test_true_color_shows_motion_as_rgb():
    from motionsheet.stack import true_color
    frames = []
    for i in range(3):
        f = np.zeros((H, W), np.uint8)
        f[20:20 + SIZE, 20 + i * 30:20 + i * 30 + SIZE] = 255
        frames.append(f)
    stack = rgb_stack([f / 255 for f in frames])
    color = np.repeat(frames[1][..., None], 3, axis=2)
    out = true_color(stack, color, frames)
    assert tuple(out[30, 30]) == (255, 0, 0)   # first position still red
    assert tuple(out[30, 90]) == (0, 0, 255)   # last position still blue
    assert tuple(out[5, 5]) == (0, 0, 0)       # untouched background: video color
