import numpy as np

def flatten_numeric(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)

def stats(values):
    x = flatten_numeric(values)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "count": 0, "min": np.nan, "max": np.nan,
            "smallest_nonzero": np.nan, "zero_fraction": np.nan,
            "min_spacing": np.nan, "median_spacing": np.nan,
            "max_spacing": np.nan
        }

    nz = finite[finite != 0]
    unique = np.unique(finite)
    diffs = np.diff(unique) if unique.size > 1 else np.array([])

    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "smallest_nonzero": float(np.min(np.abs(nz))) if nz.size else 0.0,
        "zero_fraction": float(np.mean(finite == 0)),
        "min_spacing": float(diffs.min()) if diffs.size else np.nan,
        "median_spacing": float(np.median(diffs)) if diffs.size else np.nan,
        "max_spacing": float(diffs.max()) if diffs.size else np.nan,
    }
