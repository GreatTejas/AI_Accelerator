from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def register_hooks(model):
    inputs, outputs = {}, {}
    handles = []

    types = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.GELU,
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
    )

    def make_hook(name):
        def hook(module, inp, out):
            if inp and torch.is_tensor(inp[0]):
                inputs.setdefault(name, []).append(inp[0].detach().cpu())
            if torch.is_tensor(out):
                outputs.setdefault(name, []).append(out.detach().cpu())
        return hook

    for name, module in model.named_modules():
        if name and isinstance(module, types):
            handles.append(module.register_forward_hook(make_hook(name)))

    return inputs, outputs, handles

def collect_inference(model, loader, output_dir, device):
    model.eval()
    inputs, outputs, handles = register_hooks(model)
    logits = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Inference"):
            x = x.to(device)
            out = model(x)
            logits.append(out.detach().cpu())

    for h in handles:
        h.remove()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}

    for name, vals in inputs.items():
        tensors["input__" + name.replace(".", "_")] = torch.cat(vals).numpy().astype(np.float32)
    for name, vals in outputs.items():
        tensors["activation__" + name.replace(".", "_")] = torch.cat(vals).numpy().astype(np.float32)

    tensors["output_logits"] = torch.cat(logits).numpy().astype(np.float32)

    for name, p in model.named_parameters():
        if name.endswith(".weight"):
            tensors["weight__" + name.replace(".", "_")] = p.detach().cpu().numpy().astype(np.float32)
        elif name.endswith(".bias"):
            tensors["bias__" + name.replace(".", "_")] = p.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(output_dir / "inference_tensors.npz", **tensors)
