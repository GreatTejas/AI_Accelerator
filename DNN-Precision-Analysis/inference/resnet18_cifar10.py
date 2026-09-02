from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_DIR, INFERENCE_DIR, NUM_WORKERS
from models.resnet import make_resnet18_cifar10
from common import collect_inference

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    model = make_resnet18_cifar10().to(device)
    collect_inference(model, loader, INFERENCE_DIR / "resnet18_cifar10", device)

if __name__ == "__main__":
    main()
