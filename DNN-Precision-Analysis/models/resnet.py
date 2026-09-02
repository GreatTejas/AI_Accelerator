import torch.nn as nn
from torchvision.models import resnet18

def make_resnet18_cifar10():
    model = resnet18(weights=None)
    # CIFAR-10 images are 32x32. Replace ImageNet stem.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model
