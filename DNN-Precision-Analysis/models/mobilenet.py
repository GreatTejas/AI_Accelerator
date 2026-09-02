import torch.nn as nn
from torchvision.models import mobilenet_v2

def make_mobilenetv2_cifar100():
    model = mobilenet_v2(weights=None)
    model.features[0][0] = nn.Conv2d(
        3, 32, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
    return model
