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


def test_grab_frames_returns_consecutive_frames_with_their_times():
    from pathlib import Path
    from motionsheet.cli import grab_frames
    clip = Path(__file__).parent.parent / "examples/clips/moving-square.mp4"
    frames = grab_frames(clip, 1.0, 9)
    times = [t for t, _ in frames]
    assert len(frames) == 9
    assert abs(times[0] - 1.0) < 1 / 24 + 1e-6
    assert all(abs((b - a) - 1 / 24) < 1e-3 for a, b in zip(times, times[1:]))
    assert len({img.tobytes() for _, img in frames}) > 1  # the square moved


def test_parse_time_accepts_the_frame_file_format():
    assert parse_time("00m08.40s") == 8.4
    assert parse_time("01m15s") == 75


def test_high_fps_only_for_short_windows():
    from motionsheet.cli import fps_allowed
    assert fps_allowed(12, 77)
    assert not fps_allowed(24, 77)
    assert fps_allowed(24, 6)
    assert not fps_allowed(61, 1)


def test_stamp_is_exact_to_the_millisecond():
    from motionsheet.cli import stamp
    assert stamp(36.75) == "00:36.750"
    assert stamp(36.8333) == "00:36.833"
    assert parse_time("00m36.833s") == 36.833


def test_labels_are_gray_so_only_motion_is_colored():
    from motionsheet.cli import render_sheets
    still = np.full((90, 160, 3), 128, np.uint8)
    cells = [{"kind": "stack", "times": [0, 1, 2], "holds": [1, 1, 1], "image": still}] * 4
    manifest, pages = render_sheets(cells, 2, "clip.mp4")
    px = np.asarray(pages[0]).astype(int)
    assert (px[..., 0] == px[..., 1]).all() and (px[..., 1] == px[..., 2]).all()
    assert manifest[0].startswith("#1 ")
    assert manifest[3].startswith("#4 ")


def test_no_rings_when_many_small_things_change():
    still = np.zeros((120, 200), np.uint8)
    frames = []
    for i in range(3):
        f = still.copy()
        for j in range(30):  # thirty letters, each there in one frame only
            if j % 3 == i:
                y, x = 10 + 20 * (j // 10), 10 + 18 * (j % 10)
                f[y:y + 4, x:x + 4] = 255
        frames.append(f)
    img = rgb_stack([f / 255 for f in frames])
    assert (draw_rings(img, frames) == img).all()


def test_cells_with_no_visible_color_are_dropped():
    from motionsheet.cli import build_cells
    still = np.full((60, 80, 3), 100, np.uint8)
    moved = still.copy()
    moved[20:30, 20:30] = 250
    frames = [(t, still) for t in range(3)] + [(3, still), (4, moved), (5, still)]
    cells, dropped = build_cells(frames, [(0, 6)], rings=False, color=False)
    assert [c["times"] for c in cells] == [[3, 4, 5]]
    assert dropped == 1


def test_still_video_exits_with_a_message_instead_of_crashing(tmp_path, monkeypatch):
    import pytest
    from motionsheet.cli import main
    monkeypatch.setattr("sys.argv", ["motionsheet", "examples/clips/static-noise.mp4", "-o", str(tmp_path)])
    with pytest.raises(SystemExit, match="no visible change"):
        main()


def _square(x):
    f = np.zeros((120, 200), np.uint8)
    if x is not None:
        f[50:60, x:x + 10] = 255
    return f


def test_change_finder_names_a_blink():
    from motionsheet.cli import describe_changes
    grays = [_square(None), _square(90), _square(None)]  # there in the middle only
    lines = describe_changes(grays, [1.0, 1.5, 2.0], None)
    assert len(lines) == 1
    assert "blink" in lines[0] and "middle-center" in lines[0]


def test_change_finder_names_motion_and_the_gap_before_the_cell():
    from motionsheet.cli import describe_changes
    grays = [_square(20), _square(90), _square(160)]
    lines = describe_changes(grays, [1.0, 1.5, 2.0], prev=_square(None))
    assert any("00:01.000 and 00:01.500" in l or "changed at 00:01.500" in l for l in lines)
    assert any(l.startswith("before this cell") for l in lines)


def test_change_finder_says_whole_picture_instead_of_listing():
    from motionsheet.cli import describe_changes
    rng = np.random.default_rng(1)
    grays = [(rng.random((120, 200)) * 255).astype(np.uint8) for _ in range(3)]
    assert describe_changes(grays, [0, 1, 2], None) == ["most of the picture changed"]
