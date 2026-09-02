import torch.nn as nn

COMPUTE_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.Linear,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.GELU,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
)

def is_compute_module(module):
    return isinstance(module, COMPUTE_TYPES)
