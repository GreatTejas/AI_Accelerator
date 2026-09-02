import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    for script in ["resnet18_cifar10.py", "mobilenetv2_cifar100.py", "lenet5_mnist.py"]:
        subprocess.run([sys.executable, str(base / script)], check=True)

if __name__ == "__main__":
    main()
