import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motionsheet.stack import rgb_stack

INSTALL_FFMPEG = """ffmpeg not found. Install it:
  macOS:   brew install ffmpeg
  Ubuntu:  sudo apt install ffmpeg
  Windows: winget install ffmpeg"""

SHEET_MAX = 1536
PAD = 8
TEXT_H = 30
HOLD_FULL = 1.0
LEGEND_H = 32
LEGEND = [((255, 0, 0), "red = frame 1"), ((0, 255, 0), "green = frame 2"),
          ((0, 0, 255), "blue = frame 3")]
# Pure blue text is unreadable on the dark sheet, so stamps use lighter tints.
STAMP_COLORS = [(255, 90, 90), (90, 255, 90), (110, 150, 255)]


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def detect_cuts(video, duration):
    """Shot ranges [(start, end)] in seconds."""
    from scenedetect import ContentDetector, detect
    # PyAV backend: the OpenCV backend crashes on some phone video (NaN timecode).
    scenes = detect(str(video), ContentDetector(), backend="pyav")
    if not scenes:
        return [(0.0, duration)]
    return [(a.get_seconds(), b.get_seconds()) for a, b in scenes]


def changed_frames(video, fps, width, threshold, tmp):
    """Decode at `fps` and drop near-duplicates with ffmpeg's mpdecimate.

    Camera shake is kept, not stabilized: it is part of the footage.
    Returns [(time, rgb uint8 array)].
    """
    vf = (f"fps={fps},scale={width}:-2,"
          f"mpdecimate=hi={threshold}:lo={threshold // 3}:frac=0.5,showinfo")
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video), "-an", "-vf", vf,
         "-fps_mode", "vfr", str(tmp / "f_%05d.png")],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{r.stderr[-800:]}")
    times = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", r.stderr)]
    files = sorted(tmp.glob("f_*.png"))
    return [(t, np.asarray(Image.open(f).convert("RGB")))
            for t, f in zip(times, files)]


def build_cells(frames, shots):
    """Kept frames in runs of 3 -> RGB stack cells; a plain frame marks each cut."""
    cells = []
    for n, (start, end) in enumerate(shots):
        in_shot = [f for f in frames if start <= f[0] < end]
        if not in_shot:
            continue
        if n > 0:
            cells.append({"kind": "cut", "times": [in_shot[0][0]],
                          "image": in_shot[0][1]})
        # How long each kept frame stood before the next change (or the cut).
        holds = [b[0] - a[0] for a, b in zip(in_shot, in_shot[1:])]
        holds.append(end - in_shot[-1][0])
        for i in range(0, len(in_shot), 3):
            g = in_shot[i:i + 3]
            grays = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
                     for _, img in g]
            # A shot's last run can come up short; repeating its final frame
            # means "no further change", which reads correctly in the stack.
            grays += [grays[-1]] * (3 - len(grays))
            cells.append({"kind": "stack", "times": [t for t, _ in g],
                          "holds": holds[i:i + 3], "image": rgb_stack(grays)})
    return cells


def stamp(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def legend(width, font):
    bar = Image.new("RGB", (width, LEGEND_H), (24, 24, 24))
    d = ImageDraw.Draw(bar)
    sw = (width - 2 * PAD) // 3
    for i, (color, label) in enumerate(LEGEND):
        x = PAD + i * sw
        d.rectangle([x, 6, x + sw - 4, LEGEND_H - 6], fill=color)
        d.text((x + 6, 9), label, fill=(0, 0, 0) if i < 2 else (255, 255, 255),
               font=font)
    return bar


def render_sheets(cells, cols, out):
    font = ImageFont.load_default(size=13)
    h, w = cells[0]["image"].shape[:2]
    img_w = (SHEET_MAX - PAD * (cols + 1)) // cols
    img_h = round(img_w * h / w)
    cell_h = img_h + TEXT_H
    rows = max(1, (SHEET_MAX - LEGEND_H - PAD) // (cell_h + PAD))
    sheet_w = cols * img_w + PAD * (cols + 1)
    per_sheet = cols * rows
    manifest = []
    for s, first in enumerate(range(0, len(cells), per_sheet), 1):
        chunk = cells[first:first + per_sheet]
        used_rows = (len(chunk) + cols - 1) // cols
        sheet_h = used_rows * (cell_h + PAD) + PAD + LEGEND_H
        sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 16))
        d = ImageDraw.Draw(sheet)
        for i, c in enumerate(chunk):
            x = PAD + (i % cols) * (img_w + PAD)
            y = PAD + (i // cols) * (cell_h + PAD)
            sheet.paste(Image.fromarray(c["image"]).resize((img_w, img_h)), (x, y))
            if c["kind"] == "cut":
                d.text((x + 2, y + img_h + 3), f"CUT {stamp(c['times'][0])}",
                       fill=(255, 255, 255), font=font)
                manifest.append(f"sheet {s:03d}, cell {i + 1}: CUT at "
                                f"{stamp(c['times'][0])} (plain frame)")
                continue
            slot = img_w // 3
            for k, (t, hold, color) in enumerate(zip(c["times"], c["holds"],
                                                     STAMP_COLORS)):
                tx = x + 2 + k * slot
                d.text((tx, y + img_h + 3), stamp(t), fill=color, font=font)
                # Bar length = how long this frame held; full slot = HOLD_FULL s.
                bar = max(2, round(min(hold / HOLD_FULL, 1) * (slot - 8)))
                d.rectangle([tx, y + img_h + 20, tx + bar, y + img_h + 24],
                            fill=color)
            manifest.append(f"sheet {s:03d}, cell {i + 1}: "
                            + ", ".join(f"{name} {stamp(t)} (held {h:.2f}s)"
                                        for t, h, name in zip(c["times"], c["holds"],
                                                              ("red", "green", "blue"))))
        sheet.paste(legend(sheet_w, font), (0, sheet_h - LEGEND_H))
        sheet.save(out / f"sheet-{s:03d}.png")
    return manifest


def main():
    ap = argparse.ArgumentParser(
        prog="motionsheet",
        description="Turn a video into contact sheets that show motion.")
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="output folder "
                    "(default: <video name>-motionsheet next to the video)")
    ap.add_argument("--fps", type=float, default=12, help="decode rate (12)")
    ap.add_argument("--width", type=int, default=480, help="working width (480)")
    ap.add_argument("--threshold", type=int, default=12000,
                    help="near-duplicate cutoff for ffmpeg mpdecimate; higher "
                         "drops more frames (12000)")
    ap.add_argument("--cols", type=int, default=5, help="cells per row (5)")
    ap.add_argument("--keep-frames", action="store_true",
                    help="also save every kept frame, named by timestamp")
    args = ap.parse_args()

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit(INSTALL_FFMPEG)
    if not args.video.is_file():
        sys.exit(f"no such file: {args.video}")

    out = args.out or args.video.with_name(f"{args.video.stem}-motionsheet")
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    duration = probe(args.video)
    shots = detect_cuts(args.video, duration)
    with tempfile.TemporaryDirectory() as tmp:
        frames = changed_frames(args.video, args.fps, args.width,
                                args.threshold, Path(tmp))
    if not frames:
        sys.exit("no frames decoded")
    cells = build_cells(frames, shots)
    manifest = render_sheets(cells, args.cols, out / "sheets")

    if args.keep_frames:
        (out / "frames").mkdir(exist_ok=True)
        for t, img in frames:
            Image.fromarray(img).save(out / "frames" / f"{stamp(t).replace(':', 'm')}s.png")

    (out / "MANIFEST.md").write_text(
        f"# {args.video.name}\n\n{len(shots)} shots, {len(frames)} changed frames, "
        f"{len(cells)} cells\n\n" + "\n".join(f"- {m}" for m in manifest) + "\n")
    print(f"{len(cells)} cells -> {out}")
