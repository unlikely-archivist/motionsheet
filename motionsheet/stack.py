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
