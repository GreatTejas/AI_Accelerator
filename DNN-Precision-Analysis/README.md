# DNN Numerical Range and Precision Analysis

This project analyzes FP32 numerical range and precision during training and inference for:

- ResNet-18 on CIFAR-10 (training + inference)
- MobileNetV2 on CIFAR-100 (inference)
- LeNet-5 on MNIST (inference)

The analysis collects inputs, weights, activations, gradients, weight updates, biases and output logits as required by the assignment.

## Important

The project does **not** train or run inference using FP16/FP8/INT16/INT8. Instead, it performs numerical representation/quantization simulations on the collected FP32 values.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Training

Train ResNet-18 on CIFAR-10:

```bash
python training/train_resnet18.py --epochs 100 --batch-size 128
```

The default checkpoints are collected at approximately the beginning, middle and end of training.

For a quick test:

```bash
python training/train_resnet18.py --epochs 3 --batch-size 128
```

## Inference

Run all three inference experiments:

```bash
python inference/run_all.py --batch-size 128
```

Individual experiments:

```bash
python inference/resnet18_cifar10.py
python inference/mobilenetv2_cifar100.py
python inference/lenet5_mnist.py
```

## Analysis

After data collection:

```bash
python analysis/run_analysis.py
```

Results are written under `results/`.

## Output

The analysis creates:

- CSV summary statistics
- Full-range distribution plots
- Magnified-around-zero plots
- FP16 simulation error
- FP8 E4M3/E5M2 simulation error
- INT8 and INT16 symmetric per-tensor quantization error
- Layer-wise precision recommendations

The generated CSV files can be used directly to prepare the assignment report.

## Notes on precision

INT8/INT16 use symmetric scaling:

q = round(x / scale)

x_hat = q * scale

where:

scale = max(abs(x)) / max_integer

FP16 uses NumPy float16 conversion.

FP8 E4M3 and E5M2 are simulated in software so the experiment does not require FP8 hardware.
