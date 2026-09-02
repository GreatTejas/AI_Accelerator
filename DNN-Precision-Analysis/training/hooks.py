from collections import defaultdict
import torch
import torch.nn as nn

class ActivationCollector:
    def __init__(self, model, collect_types=None):
        self.model = model
        self.handles = []
        self.inputs = defaultdict(list)
        self.outputs = defaultdict(list)
        self.collect_types = collect_types or (
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.Linear,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.GELU,
            nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
            nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
            nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
        )

    def register(self):
        for name, module in self.model.named_modules():
            if name and isinstance(module, self.collect_types):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inp, out):
            if inp and torch.is_tensor(inp[0]):
                self.inputs[name].append(inp[0].detach().cpu())
            if torch.is_tensor(out):
                self.outputs[name].append(out.detach().cpu())
        return hook

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
