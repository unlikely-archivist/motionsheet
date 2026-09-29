---
name: motionsheet
description: Watch a video file by turning it into contact sheets that show motion, not just still frames. Use when the user references a video file (.mp4, .mov, .webm, .mkv, etc.) or asks what happens, moves, or changes in a clip, screen recording, or footage.
---

# motionsheet

`motionsheet` turns a video into a PDF of contact sheets. Each cell overlays
three moments in red, green, and blue, so motion shows as colored copies and
still areas stay gray.

## Workflow: preview, overview, notes, zoom

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

3. **Notes.** Before answering anything, write `OUT/NOTES.md`: one line per
   event, with its timestamp, in plain words ("0:04 menu opens on the cat"),
   not colors. Mark anything unclear with `?`. Answer later questions from
   the notes; they survive when the sheets drop out of context.

4. **Zoom.** When the notes don't settle a question, look closer. Every
   request names exact times in the frame file format, `MMmSS.SSs`
   (`00m08.40s`), and picks one of two kinds:
   - **Full-resolution frames of one moment**, to read text, menus, or small
     objects: give the moment and how many frames (1-9, consecutive, none
     skipped; at 60fps, 9 frames is 0.15s).
     ```
     motionsheet VIDEO --frames 00m08.40s --count 9 -o OUT-frames
     ```
   - **More fps over a stretch**, to follow fast motion: give the start, the
     end, and the fps. Above 12fps the stretch can be at most 10s, and the
     fps at most 60. Use `--cols 2` here, never on the overview (it doubles
     the pages).
     ```
     motionsheet VIDEO --start 00m40.00s --end 00m46.00s --fps 24 --cols 2 --rings -o OUT-zoom
     ```
   Update `NOTES.md` with what the zoom showed.

## Choosing settings from `--stats`

| stats say | use |
|---|---|
| most changes tiny (<1% of picture): pointer, UI, small sprites | `--rings` |
| most changes cover most of the picture: camera motion, scrolling | no `--rings` (they'd add clutter) |
| many pages expected for a long video, only the gist is needed | `--cols 5` (smaller cells, fewer pages) |
| text or small detail matters | `--frames` on that moment (see Zoom) |
| nearly every frame has a visible change (something moves the whole time: animals, people, handheld footage) | `--fps 6` for the overview, then zoom at 12 or 24fps where it matters |
| very fast action (sports, games) | `--fps 24` on a stretch of at most 10s (see Zoom) |
| a busy stretch you need to understand | `--start` / `--end` on that stretch |

Defaults already adapt to the video's shape: wide video gets 3 cells per row
at 960px working width, tall video 4 per row at 480px.

## How to read the sheets

Each page header shows the file name, page number (e.g. `page 3/20`), and the
time span. Read cells left to right, top to bottom.

Each cell overlays 3 moments: **red = first, green = second, blue = third**.
The three timestamps under the cell are those moments, left to right. Every
label (numbers, timestamps, bars) is gray, so **any color in a picture is
change**. The colors are moments, not colors in the video: never describe
something as changing color.

| you see | it means |
|---|---|
| gray or white | no change; present in all three moments |
| red, green, and blue copies side by side | it moved, red → green → blue |
| a thin colored fringe | it moved a little |
| **one solid color, no other copies near it** | it was there only in some moments, in place: it **appeared, disappeared, or blinked** |
| pure red | there first, then gone |
| yellow (red + green) | there first and second, gone by the third |
| magenta (red + blue) | there first and last, gone in the middle: it blinked |
| cyan (green + blue) | appeared after the first |
| pure blue | appeared in the last |

To find small changes (a blink, a pointer, a new line of text), scan each
cell for any color at all before reading the picture itself. Cells are
numbered (`#87`) so you can refer to them.

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
- A **plain full-color cell labeled CUT** marks a new shot; it is the one
  cell whose colors are the video's own (in a screen
  recording, usually a new page or window).

`MANIFEST.md` lists every cell with exact timestamps and hold times; use it
to cite moments precisely.
