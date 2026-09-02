import numpy as np

def fp16_error(x):
    x = np.asarray(x, dtype=np.float32)
    y = x.astype(np.float16).astype(np.float32)
    return error_metrics(x, y)

def error_metrics(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    e = np.abs(x-y)
    denom = np.abs(x)
    rel = e / np.maximum(denom, 1e-12)
    return {
        "mae": float(np.mean(e)),
        "max_abs_error": float(np.max(e)),
        "rmse": float(np.sqrt(np.mean((x-y)**2))),
        "mean_relative_error": float(np.mean(rel)),
        "max_relative_error": float(np.max(rel)),
    }

def symmetric_quantize(x, bits):
    x = np.asarray(x, dtype=np.float32)
    qmax = (2 ** (bits-1)) - 1
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    if amax == 0:
        return np.zeros_like(x), 1.0
    scale = amax / qmax
    q = np.clip(np.rint(x / scale), -qmax, qmax)
    return q, scale

def int_error(x, bits):
    q, scale = symmetric_quantize(x, bits)
    y = q * scale
    m = error_metrics(x, y)
    m["scale"] = float(scale)
    return m

def fp8_quantize(x, exponent_bits, mantissa_bits):
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    ax = np.abs(x)

    if exponent_bits == 4:
        bias = 7
    elif exponent_bits == 5:
        bias = 15
    else:
        raise ValueError("Only E4M3 and E5M2 are supported.")

    max_exp_field = (2 ** exponent_bits) - 2
    max_unbiased_exp = max_exp_field - bias
    min_normal_exp = 1 - bias

    out = np.zeros_like(ax)
    nonzero = ax > 0

    # Normal numbers
    vals = ax[nonzero]
    exp = np.floor(np.log2(vals))
    exp_clipped = np.clip(exp, min_normal_exp, max_unbiased_exp)
    step = 2.0 ** (exp_clipped - mantissa_bits)
    rounded = np.round(vals / step) * step

    # Underflow to zero
    rounded[exp < min_normal_exp] = 0.0

    # Overflow to largest finite representable value
    max_mantissa = 2.0 - 2.0 ** (-mantissa_bits)
    max_finite = max_mantissa * (2.0 ** max_unbiased_exp)
    rounded = np.minimum(rounded, max_finite)

    out[nonzero] = rounded
    return out * sign

def fp8_error(x, fmt="E4M3"):
    if fmt == "E4M3":
        y = fp8_quantize(x, 4, 3)
    elif fmt == "E5M2":
        y = fp8_quantize(x, 5, 2)
    else:
        raise ValueError(fmt)
    return error_metrics(x, y)
