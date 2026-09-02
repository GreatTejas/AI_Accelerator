# DNN Numerical Range and Precision Analysis

## 1. Objective

Describe the purpose of the experiment and the question of whether FP32 can be replaced by lower precision.

## 2. Experimental Setup

| Model | Dataset | Task | Representation |
|---|---|---|---|
| ResNet-18 | CIFAR-10 | Training + inference | FP32 |
| MobileNetV2 | CIFAR-100 | Inference | FP32 |
| LeNet-5 | MNIST | Inference | FP32 |

Mention hardware, PyTorch version, batch size, epochs and random seed.

## 3. Training Analysis

### 3.1 Beginning

Discuss inputs, weights, activations, gradients, weight updates and biases layer-by-layer.

### 3.2 Middle

Repeat.

### 3.3 End

Repeat.

Include full-range and zero-zoom distributions.

## 4. Inference Analysis

Discuss all three model/dataset combinations.

For each, analyze inputs, weights, activations, biases and logits.

## 5. Precision Analysis

Discuss FP16, FP8 E4M3, FP8 E5M2, INT16 and INT8.

For integer formats, explicitly describe the scaling method.

## 6. Results

Include numerical tables for minimum, maximum, smallest nonzero magnitude and spacing.

Include reconstruction/quantization errors.

## 7. Discussion

Explain which data types can safely use lower precision and which cannot.

## 8. Conclusion

State the final recommendation separately for:

- Inputs
- Weights
- Activations
- Gradients
- Weight updates
- Biases
- Output logits
