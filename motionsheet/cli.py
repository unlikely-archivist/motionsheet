import argparse
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motionsheet.stack import rgb_stack, true_color

INSTALL_FFMPEG = """ffmpeg not found. Install it:
  macOS:   brew install ffmpeg
  Ubuntu:  sudo apt install ffmpeg
  Windows: winget install ffmpeg"""

SHEET_MAX = 1536
PAD = 8
TEXT_H = 30
HOLD_FULL = 1.0
LEGEND_H = 32
HEADER_H = 24
LEGEND = ("in the pictures: red = 1st time, green = 2nd, blue = 3rd. "
          "Any color is change; labels are gray so they never look like change.")
LABEL = (220, 220, 220)
BAR = (150, 150, 150)
# Ring colors match the stack channels; pure blue is too dark on the sheet.
RING_COLORS = [(255, 90, 90), (90, 255, 90), (110, 150, 255)]
RING_DIFF = 25          # gray-level difference that counts for a ring
RING_MAX_AREA = 0.02    # regions over 2% of the frame get no ring
VISIBLE_SPREAD = 30     # channel spread that reads as color in a stack
VISIBLE_MIN_PIXELS = 20 # fewer colored pixels than this: the cell is dropped
WHOLE_PICTURE = 0.30    # changed share over which a cell is "most of the picture"
MAX_REGIONS = 5         # changed regions listed per cell
RING_MAX_COUNT = 6      # more small changes than this in a frame: no rings
PREVIEW_MAX = 1536      # long side of each --stats preview frame
PREVIEW_COUNT = 6
QUIET_GAP = 3.0         # seconds without change reported as a quiet stretch
MAX_FAST_WINDOW = 10.0  # longest window allowed with --fps above 12
MAX_FPS = 60
BUSY_BIN = 5.0          # seconds per bin when finding the busiest stretches


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def parse_time(text):
    """"75", "1:15", "0:01:15" or "01m15.00s" (the frame file format) -> seconds."""
    seconds = 0.0
    for part in text.rstrip("s").replace("m", ":").split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def detect_cuts(video, start, end):
    """Shot ranges [(start, end)] in seconds within the window."""
    from scenedetect import ContentDetector, detect
    # PyAV backend: the OpenCV backend crashes on some phone video (NaN timecode).
    scenes = detect(str(video), ContentDetector(), backend="pyav",
                    start_time=start, end_time=end)
    if not scenes:
        return [(start, end)]
    return [(a.get_seconds(), b.get_seconds()) for a, b in scenes]


def fps_allowed(fps, window):
    """Above the default rate, only short windows, so a zoom can't cost as
    much as a whole video."""
    return fps <= 12 or (fps <= MAX_FPS and window <= MAX_FAST_WINDOW)


def native_size(video):
    """Display size (w, h) of the video, after any rotation."""
    first = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(first)).size


def decode(video, fps, width, start, end):
    """Yield (time, rgb uint8 array) at `fps`, scaled to `width`, from
    `start` to `end` seconds, streamed so long videos never sit in memory
    all at once."""
    first = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True).stdout
    w, h = Image.open(io.BytesIO(first)).size
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(video),
         "-t", str(end - start), "-an",
         "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    size = w * h * 3
    i = 0
    while chunk := proc.stdout.read(size):
        if len(chunk) < size:
            break
        yield start + i / fps, np.frombuffer(chunk, np.uint8).reshape(h, w, 3)
        i += 1
    proc.wait()


def patch_changes(a, b):
    """Mean change (0-255) of every 8x8 patch between two gray frames.
    Patch-based so a small change, like a moving cursor, isn't averaged
    away by a still screen."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    h, w = (d.shape[0] // 8) * 8, (d.shape[1] // 8) * 8
    return d[:h, :w].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))


def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def changed_frames(video, fps, width, min_change, max_gap, start, end):
    """Keep a frame when any 8x8 patch differs from the last kept frame by
    more than `min_change`, and at least one every `max_gap` seconds.
    Camera shake is kept, not stabilized: it is part of the footage.
    Returns [(time, rgb uint8 array)], the number of frames decoded, and for
    each kept frame after the first, the share of the picture that changed."""
    kept, areas, decoded, last_g, last_t = [], [], 0, None, None
    for t, rgb in decode(video, fps, width, start, end):
        decoded += 1
        g = gray(rgb)
        if last_g is None:
            changed = None
        else:
            patches = patch_changes(last_g, g)
            changed = patches.max() > min_change
        if changed is None or changed or t - last_t >= max_gap:
            if changed is not None:
                areas.append(float((patches > min_change).mean()))
            kept.append((t, rgb.copy()))
            last_g, last_t = g, t
    return kept, decoded, areas


def draw_rings(stack, grays):
    """Circle each small region that appears in only one of the 3 frames,
    in that frame's color, so small movers like a pointer stand out. Large
    changes (scrolls, page loads) are skipped: they're already visible. So
    is a frame with many small changes: rings only help when there are few."""
    img = Image.fromarray(stack)
    d = ImageDraw.Draw(img)
    h, w = stack.shape[:2]
    width = max(2, w // 300)
    for k, g in enumerate(grays):
        others = [o for j, o in enumerate(grays) if j != k]
        only_here = np.min([cv2.absdiff(g, o) for o in others], axis=0)
        mask = cv2.dilate((only_here > RING_DIFF).astype(np.uint8), np.ones((5, 5)))
        n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        small = [i for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] <= RING_MAX_AREA * w * h]
        if len(small) > RING_MAX_COUNT:
            continue  # many small changes (text, scrolling): rings would bury the picture
        for i in small:
            x, y = centers[i]
            r = max(w // 60, int(max(stats[i, 2], stats[i, 3]) * 0.8))
            d.ellipse([x - r, y - r, x + r, y + r], outline=RING_COLORS[k],
                      width=width)
    return np.asarray(img)


def place(x, y, w, h):
    """Rough position name, e.g. "top-left", for a point in a w x h picture."""
    row = ("top", "middle", "bottom")[min(int(3 * y / h), 2)]
    col = ("left", "center", "right")[min(int(3 * x / w), 2)]
    return f"{row}-{col}"


def change_regions(masks):
    """Changed regions of the union of `masks`, largest first, as
    (label image, stats, centers, index) with nearby bits (letters of one
    word) merged into one region."""
    both = np.max(masks, axis=0).astype(np.uint8)
    k = max(3, both.shape[1] // 60)
    merged = cv2.dilate(both, np.ones((k, k), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(merged)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return labels, stats, centers, order


def describe_changes(grays, times, prev):
    """What changed in a cell, as text: where, how big, and when, per
    region. `prev` is the previous cell's last frame (None after a cut), so
    a change between two cells is reported too."""
    h, w = grays[0].shape
    steps = [cv2.absdiff(a, b) > RING_DIFF for a, b in zip(grays, grays[1:])]
    if np.max(steps, axis=0).mean() > WHOLE_PICTURE:
        return ["most of the picture changed"]
    back = cv2.absdiff(grays[0], grays[-1]) > RING_DIFF
    labels, stats, centers, order = change_regions(steps)
    lines = []
    for i in order[:MAX_REGIONS]:
        region = labels == i
        n = max(1, max(s[region].sum() for s in steps))
        when = [s[region].sum() > 0.2 * n for s in steps]
        x, y, bw, bh = stats[i, :4]
        where = (f"{place(*centers[i], w, h)} (x {100 * x // w}-{100 * (x + bw) // w}%, "
                 f"y {100 * y // h}-{100 * (y + bh) // h}%), "
                 f"{region.mean():.1%} of picture")
        if all(when) and back[region].sum() < 0.2 * n:
            what = (f"blink: different only at {stamp(times[1])} "
                    "(appeared and went, or passed through)")
        elif all(when):
            what = f"changed at {stamp(times[1])} and again at {stamp(times[2])}"
        else:
            what = f"changed at {stamp(times[1 if when[0] else 2])}"
        lines.append(f"{where}: {what}")
    if len(order) > MAX_REGIONS:
        lines.append(f"+{len(order) - MAX_REGIONS} smaller changes")
    if prev is not None:
        gap = cv2.absdiff(prev, grays[0]) > RING_DIFF
        if gap.mean() > WHOLE_PICTURE:
            lines.append("before this cell: most of the picture changed")
        elif gap.any():
            labels, stats, centers, order = change_regions([gap])
            lines.append("before this cell: changed at " + ", ".join(
                place(*centers[i], w, h) for i in order[:MAX_REGIONS]))
    return lines


def build_cells(frames, shots, rings, color):
    """Kept frames in runs of 3 -> RGB stack cells; a plain frame marks each
    cut. A cell with no visible color is dropped: it shows nothing."""
    cells, dropped = [], 0
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
        prev = None  # last frame of the previous cell in this shot
        for i in range(0, len(in_shot), 3):
            g = in_shot[i:i + 3]
            grays = [gray(img) for _, img in g]
            before, prev = prev, grays[-1]
            # A shot's last run can come up short; repeating its final frame
            # means "no further change", which reads correctly in the stack.
            grays += [grays[-1]] * (3 - len(grays))
            image = rgb_stack([x.astype(np.float32) / 255 for x in grays])
            spread = image.max(axis=2).astype(int) - image.min(axis=2)
            if (spread > VISIBLE_SPREAD).sum() < VISIBLE_MIN_PIXELS:
                dropped += 1  # no visible color: nothing to see in this cell
                continue
            if color:
                image = true_color(image, g[len(g) // 2][1], grays)
            if rings:
                image = draw_rings(image, grays)
            cells.append({"kind": "stack", "times": [t for t, _ in g],
                          "holds": holds[i:i + 3], "image": image,
                          "changes": describe_changes(grays, [t for t, _ in g]
                                                      + [g[-1][0]] * (3 - len(g)),
                                                      before)})
    return cells, dropped


def stamp(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"


def legend(width, font):
    bar = Image.new("RGB", (width, LEGEND_H), (24, 24, 24))
    ImageDraw.Draw(bar).text((PAD, 9), LEGEND, fill=LABEL, font=font)
    return bar


def render_sheets(cells, cols, title):
    font = ImageFont.load_default(size=13)
    h, w = cells[0]["image"].shape[:2]
    img_w = (SHEET_MAX - PAD * (cols + 1)) // cols
    img_h = round(img_w * h / w)
    cell_h = img_h + TEXT_H
    rows = max(1, (SHEET_MAX - HEADER_H - LEGEND_H - PAD) // (cell_h + PAD))
    sheet_w = cols * img_w + PAD * (cols + 1)
    per_sheet = cols * rows
    manifest, pages = [], []
    total = (len(cells) + per_sheet - 1) // per_sheet
    for s, first in enumerate(range(0, len(cells), per_sheet), 1):
        chunk = cells[first:first + per_sheet]
        used_rows = (len(chunk) + cols - 1) // cols
        sheet_h = HEADER_H + used_rows * (cell_h + PAD) + PAD + LEGEND_H
        sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 16))
        d = ImageDraw.Draw(sheet)
        for i, c in enumerate(chunk):
            num = f"#{first + i + 1}"
            x = PAD + (i % cols) * (img_w + PAD)
            y = HEADER_H + PAD + (i // cols) * (cell_h + PAD)
            sheet.paste(Image.fromarray(c["image"]).resize((img_w, img_h)), (x, y))
            # Cell number in a dark tab on the picture's top-left corner.
            tw = d.textlength(num, font=font) + 8
            d.rectangle([x, y, x + tw, y + 17], fill=(16, 16, 16))
            d.text((x + 4, y + 2), num, fill=LABEL, font=font)
            if c["kind"] == "cut":
                d.text((x + 2, y + img_h + 3), f"CUT {stamp(c['times'][0])}",
                       fill=(255, 255, 255), font=font)
                manifest.append(f"{num} (page {s}): CUT at "
                                f"{stamp(c['times'][0])} (plain frame)")
                continue
            slot = img_w // 3
            for k, (t, hold) in enumerate(zip(c["times"], c["holds"])):
                tx = x + 2 + k * slot
                d.text((tx, y + img_h + 3), stamp(t), fill=LABEL, font=font)
                # Bar length = how long this frame held; full slot = HOLD_FULL s.
                bar = max(2, round(min(hold / HOLD_FULL, 1) * (slot - 8)))
                d.rectangle([tx, y + img_h + 20, tx + bar, y + img_h + 24],
                            fill=BAR)
            manifest.append(f"{num} (page {s}): "
                            + ", ".join(f"{name} {stamp(t)} (held {h:.2f}s)"
                                        for t, h, name in zip(c["times"], c["holds"],
                                                              ("red", "green", "blue")))
                            + "".join(f"\n  - {x}" for x in c.get("changes", [])))
        span = f"{stamp(chunk[0]['times'][0])} - {stamp(chunk[-1]['times'][-1])}"
        d.text((PAD, 6), f"{title}  ·  page {s}/{total}  ·  {span}",
               fill=(230, 230, 230), font=font)
        sheet.paste(legend(sheet_w, font), (0, sheet_h - LEGEND_H))
        pages.append(sheet)
    return manifest, pages


def grab_frame(video, t):
    """One full-detail frame at `t`, fit within PREVIEW_MAX."""
    png = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={PREVIEW_MAX}:{PREVIEW_MAX}:force_original_aspect_ratio=decrease",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(png))


def grab_frames(video, t, count):
    """`count` consecutive frames from `t` at the video's own frame rate (none
    skipped), fit within PREVIEW_MAX, as [(time, image)]. Times come from the
    video itself, so variable-rate screen recordings are timed correctly."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", str(t), "-copyts", "-i", str(video),
         "-frames:v", str(count), "-an",
         "-vf", f"showinfo,scale={PREVIEW_MAX}:{PREVIEW_MAX}:"
                "force_original_aspect_ratio=decrease",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True)
    times = [float(line.split("pts_time:")[1].split()[0])
             for line in proc.stderr.decode().splitlines() if "pts_time:" in line]
    images, data = [], proc.stdout
    while data:
        img = Image.open(io.BytesIO(data))
        img.load()
        images.append(img)
        data = data[data.index(b"IEND") + 8:]
    return list(zip(times, images))


def write_stats(video, out, start, end, shots, frames, decoded, areas, size):
    """Measurements plus a few sharp frames, so a reader can choose settings
    before rendering sheets."""
    times = [t for t, _ in frames]
    gaps = [(a, b) for a, b in zip(times, times[1:] + [end]) if b - a >= QUIET_GAP]
    bins = {}
    for t in times:
        k = int((t - start) // BUSY_BIN)
        bins[k] = bins.get(k, 0) + 1
    busiest = sorted(bins, key=bins.get, reverse=True)[:3]
    if areas:
        tiny = sum(a < 0.01 for a in areas) / len(areas)
        whole = sum(a > 0.30 for a in areas) / len(areas)
        sizes = (f"size of changes: tiny (<1% of picture) {tiny:.0%}, "
                 f"local {1 - tiny - whole:.0%}, most of picture (>30%) {whole:.0%}")
    else:
        sizes = "size of changes: no changes"
    lines = [
        f"{video.name}: {stamp(start)} - {stamp(end)}, {size[0]}x{size[1]}, "
        f"{len(shots)} shot(s)",
        "cuts at: " + (", ".join(stamp(a) for a, _ in shots[1:]) or "none"),
        f"frames with a visible change: {len(frames)} of {decoded} "
        f"({len(frames) / max(decoded, 1):.0%})",
        sizes,
        "busiest stretches: " + ", ".join(
            f"{stamp(start + k * BUSY_BIN)} - {stamp(min(start + (k + 1) * BUSY_BIN, end))} "
            f"({bins[k]} changes)" for k in sorted(busiest)),
        f"quiet stretches (>= {QUIET_GAP:.0f}s, nothing changed): " + (", ".join(
            f"{stamp(a)} - {stamp(b)}" for a, b in gaps) or "none"),
    ]
    # Always the evenly spaced frames; fill the rest with shot starts.
    even = [start + f * (end - start) for f in (0.1, 0.5, 0.9)]
    cuts = [a for a, _ in shots[1:]]
    room = PREVIEW_COUNT - len(even)
    if len(cuts) > room:
        cuts = [cuts[round(i * (len(cuts) - 1) / max(room - 1, 1))] for i in range(room)]
    picks = sorted(set(even + cuts))
    for i, t in enumerate(picks, 1):
        grab_frame(video, t).save(out / f"preview-{i}.png")
    lines.append("preview frames: " + ", ".join(
        f"preview-{i}.png at {stamp(t)}" for i, t in enumerate(picks, 1)))
    text = "\n".join(lines) + "\n"
    (out / "STATS.md").write_text(text)
    print(text, end="")


def main():
    ap = argparse.ArgumentParser(
        prog="motionsheet",
        description="Turn a video into contact sheets that show motion.")
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="output folder "
                    "(default: <video name>-motionsheet next to the video)")
    ap.add_argument("--fps", type=float, default=12, help="decode rate (12)")
    ap.add_argument("--width", type=int, help="working width in px (default: "
                    "960 for wide video, 480 for tall)")
    ap.add_argument("--min-change", type=float, default=12,
                    help="keep a frame when any 8x8 patch changes by more than "
                         "this (0-255, default 12)")
    ap.add_argument("--max-gap", type=float, default=5,
                    help="keep a frame at least this often, in seconds (5)")
    ap.add_argument("--cols", type=int, help="cells per row (default: 3 for "
                    "wide video so text stays readable, 4 for tall)")
    ap.add_argument("--keep-frames", action="store_true",
                    help="also save every kept frame, named by timestamp")
    ap.add_argument("--start", help="start of the window to render, e.g. 1:15")
    ap.add_argument("--end", help="end of the window to render, e.g. 1:30")
    ap.add_argument("--true-color", action="store_true",
                    help="show still areas in the video's own color; only "
                         "changed areas use the red/green/blue overlay")
    ap.add_argument("--rings", action="store_true",
                    help="circle small changes (pointer, small sprites) in the "
                         "color of the frame they appear in")
    ap.add_argument("--stats", action="store_true",
                    help="don't render sheets; print measurements and save a "
                         "few full-detail preview frames, to help choose settings")
    ap.add_argument("--no-sheets", action="store_true",
                    help="write only MANIFEST.md (cell times and the changes in "
                         "each cell as text), no PDF")
    ap.add_argument("--frames", metavar="TIME",
                    help="don't render sheets; save full-detail consecutive frames "
                         "from TIME, none skipped, to look closely at one moment")
    ap.add_argument("--count", type=int, default=9, choices=range(1, 10),
                    metavar="1-9", help="how many frames --frames saves (9)")
    args = ap.parse_args()

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit(INSTALL_FFMPEG)
    if not args.video.is_file():
        sys.exit(f"no such file: {args.video}")

    out = args.out or args.video.with_name(f"{args.video.stem}-motionsheet")
    out.mkdir(parents=True, exist_ok=True)
    w, h = native_size(args.video)
    wide = w >= h
    width = args.width or (960 if wide else 480)
    cols = args.cols or (3 if wide else 4)

    duration = probe(args.video)
    start = parse_time(args.start) if args.start else 0.0
    end = min(parse_time(args.end), duration) if args.end else duration
    if not 0 <= start < end:
        sys.exit(f"bad window: {args.start} to {args.end} (video is {stamp(duration)})")

    if args.frames:
        for t, img in grab_frames(args.video, parse_time(args.frames), args.count):
            name = f"frame-{stamp(t).replace(':', 'm')}s.png"
            img.save(out / name)
            print(name)
        return

    if not fps_allowed(args.fps, end - start):
        sys.exit(f"--fps above 12 is only for a window of at most "
                 f"{MAX_FAST_WINDOW:.0f}s (use --start and --end), and at most {MAX_FPS}")

    shots = detect_cuts(args.video, start, end)
    frames, decoded, areas = changed_frames(args.video, args.fps, width,
                                            args.min_change, args.max_gap, start, end)
    if not frames:
        sys.exit("no frames decoded")
    if args.stats:
        write_stats(args.video, out, start, end, shots, frames, decoded, areas, (w, h))
        return
    cells, dropped = build_cells(frames, shots, args.rings, args.true_color)
    if not cells:
        sys.exit(f"no visible change in {args.video.name}: nothing to show "
                 f"(try a lower --min-change)")
    manifest, pages = render_sheets(cells, cols, args.video.name)
    if not args.no_sheets:
        pages[0].save(out / "sheets.pdf", save_all=True, append_images=pages[1:])

    if args.keep_frames:
        (out / "frames").mkdir(exist_ok=True)
        for t, img in frames:
            Image.fromarray(img).save(out / "frames" / f"{stamp(t).replace(':', 'm')}s.png")

    (out / "MANIFEST.md").write_text(
        f"# {args.video.name}\n\n{len(shots)} shots, {len(frames)} changed frames, "
        f"{len(cells)} cells ({dropped} more had no visible change and were dropped)\n\n" + "\n".join(f"- {m}" for m in manifest) + "\n")
    print(f"{len(cells)} cells -> {out}")
