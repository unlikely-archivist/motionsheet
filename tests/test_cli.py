import numpy as np

from motionsheet.cli import draw_rings, parse_time
from motionsheet.stack import rgb_stack


def test_parse_time():
    assert parse_time("75") == 75
    assert parse_time("1:15") == 75
    assert parse_time("0:01:15.5") == 75.5


def test_rings_mark_a_small_mover_and_leave_still_frames_alone():
    still = np.zeros((120, 200), np.uint8)
    img = rgb_stack([still / 255] * 3)
    assert (draw_rings(img, [still] * 3) == img).all()

    moving = []
    for i in range(3):
        f = still.copy()
        f[50:56, 40 + i * 50:46 + i * 50] = 255
        moving.append(f)
    img = rgb_stack([f / 255 for f in moving])
    ringed = draw_rings(img, moving)
    assert (ringed != img).any(axis=2).sum() > 100  # rings were drawn
