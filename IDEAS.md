# Ideas (not in v1)

v1 targets screen recordings. These are for later.

## Filmed video mode

For handheld or moving-camera footage, where something changes in every frame.

- Align the 3 moments of a cell to the middle one (`cv2.findTransformECC` with
  `MOTION_AFFINE`, or ORB + `findHomography`, so zoom is handled). Keep the
  transforms.
- What still changes after alignment is the subject. Color only that; show
  the rest as the middle frame in gray, so camera motion stops smearing the
  cell.
- The transforms are the camera motion: report them as text per cell
  ("handheld, drift left ~8%").
- Choose the 3 moments by amount of motion, not fixed time: fast action gets
  close moments, slow stretches get wide ones.

Measured so far: on testvideo3 (opossum), translation-only alignment cut the
share of the picture changing per frame from 21% to 3% (the remainder is the
animal). On testvideo (arm wrestling) only 27% to 20%: parallax, which no 2D
alignment fixes, so the mask will be rough there.

## Cross-cell matching (change finder)

Match each changed region with the same box in nearby cells. A box that
changes in nearly every cell is one line ("keeps changing: animation") and is
left out of the cells; a box that changes, stops, and changes again is
reported as "blinks at … in #106, #108, #109". Seen on testvideo2: the side
animation fills most of the 5 region slots per cell and buries the blink.

## Other

- Sound and transcription (needs a model download).
- Ticking clocks and looping animations keep adding frames; detect and
  ignore a region that changes on a fixed cycle.
- Paper-shaped PDF pages, for people printing sheets (more pixels than models
  need).
