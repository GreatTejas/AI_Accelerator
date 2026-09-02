from pathlib import Path
import subprocess
import sys
from config import RESULTS_DIR

def main():
    targets = [
        (RESULTS_DIR / "training", RESULTS_DIR / "analysis" / "training"),
        (RESULTS_DIR / "inference", RESULTS_DIR / "analysis" / "inference"),
    ]
    here = Path(__file__).resolve().parent
    script = here / "analyze_arrays.py"

    for inp, out in targets:
        if inp.exists():
            print(f"Analyzing {inp}")
            subprocess.run(
                [sys.executable, str(script), "--input", str(inp), "--output", str(out)],
                check=True,
            )

if __name__ == "__main__":
    main()
