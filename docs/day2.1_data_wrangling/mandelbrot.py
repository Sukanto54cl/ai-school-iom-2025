import numpy as np

HEIGHT = 512
WIDTH = 512
N_LEVELS = 120000#_000
CENTER = (-0.743645857047151, 0.13182592380533)
SCALE0 = 3.0
ZOOM_PER_LEVEL = 1.00025
MAX_ITER = 256

def to_uint8(arr, max_val=MAX_ITER):
    a = np.asarray(arr, dtype=np.float32)
    a = np.clip(a, 0, float(max_val))
    a = (255.0 * a / float(max_val)).astype(np.uint8)
    return a

def mandelbrot_array(height=HEIGHT, width=WIDTH, center=CENTER, scale=SCALE0, max_iter=MAX_ITER, dtype=np.uint16):
    cx, cy = center
    aspect = width / height
    x = np.linspace(cx - (scale * aspect) / 2.0, cx + (scale * aspect) / 2.0, width, dtype=np.float64)
    y = np.linspace(cy - scale / 2.0, cy + scale / 2.0, height, dtype=np.float64)
    C = x[None, :] + 1j * y[:, None]
    Z = np.zeros_like(C)
    counts = np.zeros(C.shape, dtype=np.int32)
    mask = np.ones(C.shape, dtype=bool)
    for k in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = (np.abs(Z) > 2.0) & mask
        counts[escaped] = k
        mask &= ~escaped
        if not mask.any():
            break
    counts[mask] = max_iter
    return counts.astype(dtype, copy=False)