from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

TRAINING_DIR = RESULTS_DIR / "training"
INFERENCE_DIR = RESULTS_DIR / "inference"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

for p in [DATA_DIR, CHECKPOINT_DIR, TRAINING_DIR, INFERENCE_DIR, ANALYSIS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_WORKERS = 2
