from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_DIR, INFERENCE_DIR, NUM_WORKERS
from models.mobilenet import make_mobilenetv2_cifar100
from common import collect_inference

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    ds = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    model = make_mobilenetv2_cifar100().to(device)
    collect_inference(model, loader, INFERENCE_DIR / "mobilenetv2_cifar100", device)

if __name__ == "__main__":
    main()
