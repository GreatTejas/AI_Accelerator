import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, CHECKPOINT_DIR, TRAINING_DIR, NUM_WORKERS
from models.resnet import make_resnet18_cifar10
from utils.seed import set_seed
from hooks import ActivationCollector

def save_npz(stage_dir, model, collector, gradients, updates):
    stage_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}

    # Weights and biases.
    for name, p in model.named_parameters():
        arr = p.detach().cpu().numpy().astype(np.float32)
        if name.endswith(".weight"):
            tensors["weight__" + name.replace(".", "_")] = arr
        elif name.endswith(".bias"):
            tensors["bias__" + name.replace(".", "_")] = arr

    # Activations and inputs collected from batches.
    for name, values in collector.inputs.items():
        if values:
            tensors["input__" + name.replace(".", "_")] = torch.cat(values).numpy().astype(np.float32)
    for name, values in collector.outputs.items():
        if values:
            tensors["activation__" + name.replace(".", "_")] = torch.cat(values).numpy().astype(np.float32)

    for name, arr in gradients.items():
        tensors["gradient__" + name.replace(".", "_")] = arr
    for name, arr in updates.items():
        tensors["weight_update__" + name.replace(".", "_")] = arr

    np.savez_compressed(stage_dir / "training_tensors.npz", **tensors)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = make_resnet18_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[max(1, int(args.epochs*0.5)), max(1, int(args.epochs*0.75))],
        gamma=0.1
    )

    collector = ActivationCollector(model)
    collector.register()

    # Stages: beginning, middle, end.
    stages = {1: "beginning", max(1, args.epochs // 2): "middle", args.epochs: "end"}
    previous = {name: p.detach().clone() for name, p in model.named_parameters()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        collector.clear()
        gradients = {}

        # Collect one representative training pass for the stage.
        collect_stage = epoch in stages

        if collect_stage:
            # Keep enough data for distributions without exploding RAM.
            collector.clear()

        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            if collect_stage and batch_idx < 8:
                for name, p in model.named_parameters():
                    if p.grad is not None and name not in gradients:
                        gradients[name] = p.grad.detach().cpu().numpy().astype(np.float32)

            optimizer.step()

            if collect_stage and batch_idx < 8:
                # Weight updates after optimizer.step().
                pass

        updates = {}
        if collect_stage:
            for name, p in model.named_parameters():
                new = p.detach().clone()
                updates[name] = (new - previous[name]).cpu().numpy().astype(np.float32)
                previous[name] = new

            stage = stages[epoch]
            save_npz(TRAINING_DIR / stage, model, collector, gradients, updates)
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"resnet18_{stage}.pth")
            print("Saved stage:", stage)

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: test accuracy={acc:.4f}")

    collector.remove()

if __name__ == "__main__":
    main()
