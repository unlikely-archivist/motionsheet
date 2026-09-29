import numpy as np


def rgb_stack(frames):
    """Three grayscale frames in [0, 1] -> (H, W, 3) uint8.

    Frame 1 becomes the red channel, frame 2 green, frame 3 blue. Anything
    that holds still is equal in all three and comes out gray; anything that
    moved shows the color of the frame(s) it was in.
    """
    if len(frames) != 3:
        raise ValueError(f"need exactly 3 frames, got {len(frames)}")
    stack = np.stack([np.asarray(f, dtype=np.float64) for f in frames], axis=-1)
    return (np.clip(stack, 0, 1) * 255).round().astype(np.uint8)


def true_color(stack, color_frame, grays, soft=(8, 28)):
    """Show still areas in the video's own color and only changed areas as
    the RGB stack, so the picture looks like the footage and color fringes
    appear only where something moved.

    stack: (H, W, 3) uint8 from rgb_stack. color_frame: (H, W, 3) uint8, the
    middle frame. grays: the three uint8 grayscale frames. Pixels whose
    brightness differs by less than soft[0] across the three frames are pure
    video color; by more than soft[1], pure stack; blended in between.
    """
    g = np.stack(grays).astype(np.int16)
    spread = (g.max(axis=0) - g.min(axis=0)).astype(np.float32)
    lo, hi = soft
    mask = np.clip((spread - lo) / (hi - lo), 0, 1)[..., None]
    out = color_frame.astype(np.float32) * (1 - mask) + stack.astype(np.float32) * mask
    return out.round().astype(np.uint8)
