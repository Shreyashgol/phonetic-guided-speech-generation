"""Duration metrics. Durations are in mel frames (hop 256 @ 22,050 Hz = 11.61 ms per frame)."""
import numpy as np

FRAME_MS = 256 / 22050 * 1000


def summarize(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_res, ss_tot = (err ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return {
        'mae_frames': float(np.abs(err).mean()),
        'mae_ms': float(np.abs(err).mean() * FRAME_MS),
        'rmse_frames': float(np.sqrt((err ** 2).mean())),
        'rmse_ms': float(np.sqrt((err ** 2).mean()) * FRAME_MS),
        'r2': float(1 - ss_res / ss_tot),
        'bias_frames': float(err.mean()),
        'n': int(len(y)),
    }
