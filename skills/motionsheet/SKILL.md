---
name: motionsheet
description: Watch a video file by turning it into contact sheets that show motion, not just still frames. Use when the user references a video file (.mp4, .mov, .webm, .mkv, etc.) or asks what happens, moves, or changes in a clip, screen recording, or footage.
---

# motionsheet

`motionsheet` turns a video into a PDF of contact sheets. Each cell overlays
three moments in red, green, and blue, so motion shows as colored copies and
still areas stay gray.

## Workflow: preview, overview, zoom

1. **Preview (cheap, text first).**
   ```
   motionsheet VIDEO --stats -o OUT
   ```
   Prints measurements and saves `OUT/preview-*.png` (a few full-detail
   frames). Read the text, then look at the preview frames to learn what kind
   of video it is (screen recording, phone footage, film, game).

2. **Overview.** Choose settings from the stats and preview, then:
   ```
   motionsheet VIDEO [flags] -o OUT
   ```
   Read `OUT/sheets.pdf` (at most 20 pages per read; use the `pages`
   parameter for more) and `OUT/MANIFEST.md`.

3. **Zoom.** When a moment matters, rerun on just that window with more
   detail:
   ```
   motionsheet VIDEO --start 0:40 --end 0:46 --rings -o OUT-zoom
   ```

## Choosing settings from `--stats`

| stats say | use |
|---|---|
| most changes tiny (<1% of picture): pointer, UI, small sprites | `--rings` |
| most changes cover most of the picture: camera motion, scrolling | no `--rings` (they'd add clutter) |
| many pages expected for a long video, only the gist is needed | `--cols 5` (smaller cells, fewer pages) |
| text or small detail matters | `--cols 2` (bigger cells, more pages) |
| very fast action (sports, games) | `--fps 24` |
| a busy stretch you need to understand | `--start` / `--end` on that stretch |

Defaults already adapt to the video's shape: wide video gets 3 cells per row
at 960px working width, tall video 4 per row at 480px.

## How to read the sheets

Each page header shows the file name, page number (e.g. `page 3/20`), and the
time span. Read cells left to right, top to bottom.

Each cell overlays 3 moments: **red = first, green = second, blue = third**.
The timestamps under the cell use the same colors.

- **Gray, white, or normal-looking areas did not change.** Only change is
  colored. Do not describe the colors as a visual effect in the video.
- **Colored copies are motion.** An object's red copy is where it was first,
  green next, blue where it ended up, so red → green → blue is the direction
  of movement.
- **Yellow (red+green), cyan (green+blue), or white** means positions
  overlap; a thin fringe means a small movement.
- **Colored edges across the whole picture** mean the camera or the whole
  screen moved (pan, shake, scroll), not one object.
- **Rings** (with `--rings`) circle small changes in the color of the frame
  they appear in: red ring = where it was first, blue ring = where it ended
  up. They mark where something changed, not what it is.

Timing:

- Frames with no visible change are skipped, so a jump in timestamps between
  cells means nothing changed in between. A frame is still kept at least
  every 5 seconds (`--max-gap`), so long waits show as sparse, similar cells.
- The bar under each timestamp shows how long that moment held before the
  next change (full width = 1 second). Long bars = pauses, short bars = fast
  action.
- A **plain full-color cell labeled CUT** marks a new shot (in a screen
  recording, usually a new page or window).

`MANIFEST.md` lists every cell with exact timestamps and hold times; use it
to cite moments precisely.
