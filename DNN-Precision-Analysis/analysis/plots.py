from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def distribution_plots(values, title, out_dir, stem, bins=200, zoom_quantile=0.99):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    # To keep plots responsive for very large datasets, use a deterministic sample.
    if x.size > 500_000:
        rng = np.random.default_rng(42)
        x = rng.choice(x, 500_000, replace=False)

    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=bins, density=True)
    plt.xlabel("Numerical value")
    plt.ylabel("Probability density")
    plt.title(title + " - Full numerical range")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_full.png", dpi=160)
    plt.close()

    lim = np.quantile(np.abs(x), zoom_quantile)
    if not np.isfinite(lim) or lim <= 0:
        lim = max(float(np.max(np.abs(x))), 1e-6)
    lim *= 1.05

    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=bins, range=(-lim, lim), density=True)
    plt.xlabel("Numerical value")
    plt.ylabel("Probability density")
    plt.title(title + " - Magnified around zero")
    plt.xlim(-lim, lim)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_zero_zoom.png", dpi=160)
    plt.close()
