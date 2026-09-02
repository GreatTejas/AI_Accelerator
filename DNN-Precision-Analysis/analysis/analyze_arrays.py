import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from statistics import stats
from quantization import fp16_error, fp8_error, int_error
from plots import distribution_plots

def analyze_npz(path, output_root):
    data = np.load(path, allow_pickle=False)
    rows = []

    for key in data.files:
        x = data[key]
        s = stats(x)

        q = {}
        q.update({f"fp16_{k}": v for k, v in fp16_error(x).items()})
        q.update({f"fp8_e4m3_{k}": v for k, v in fp8_error(x, "E4M3").items()})
        q.update({f"fp8_e5m2_{k}": v for k, v in fp8_error(x, "E5M2").items()})
        q.update({f"int8_{k}": v for k, v in int_error(x, 8).items()})
        q.update({f"int16_{k}": v for k, v in int_error(x, 16).items()})

        row = {"source": str(path), "tensor": key, **s, **q}
        rows.append(row)

        plot_dir = Path(output_root) / "plots"
        distribution_plots(
            x,
            title=f"{path.stem} / {key}",
            out_dir=plot_dir,
            stem=f"{path.stem}_{key}".replace("/", "_").replace(".", "_")
        )

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory containing NPZ files")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(root.rglob("*.npz")):
        rows.extend(analyze_npz(path, out))

    pd.DataFrame(rows).to_csv(out / "precision_analysis.csv", index=False)
    print(f"Saved {len(rows)} tensor analyses to {out}")

if __name__ == "__main__":
    main()
