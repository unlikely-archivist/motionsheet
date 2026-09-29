"""Make examples/clips/camera-pan.mp4: a still scene the camera pans across,
so every edge moves the same way and nothing in the scene moves on its own."""
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS, SPEED = 480, 270, 24, 5, 3  # SPEED in px per frame

# A scene wider than the frame: gray wall, dark doorway, a picture frame, a lamp.
scene = np.full((H, W + SPEED * FPS * SECONDS, 3), 150, np.uint8)
cv2.rectangle(scene, (120, 60), (200, 270), (60, 60, 60), -1)
cv2.rectangle(scene, (300, 70), (400, 150), (230, 230, 230), 6)
cv2.circle(scene, (560, 90), 35, (240, 240, 240), -1)
cv2.rectangle(scene, (553, 125), (567, 270), (90, 90, 90), -1)
cv2.rectangle(scene, (680, 40), (800, 200), (200, 200, 200), -1)

out = Path(__file__).parent / "clips" / "camera-pan.mp4"
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", str(out)],
    stdin=subprocess.PIPE)
for i in range(FPS * SECONDS):
    ff.stdin.write(scene[:, i * SPEED:i * SPEED + W].tobytes())
ff.stdin.close()
ff.wait()
print(out)
