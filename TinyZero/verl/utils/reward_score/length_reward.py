# adapted/created for RL COURSE at MIMUW

# compute the R_i = correcntess - \lambda * (len / cap_len)

import os


def get_length_config():
    return {
        "lam": float(os.environ.get("LENGTH_LAMBDA", "0.0")),
        "variant": os.environ.get("LENGTH_VARIANT", "plain"),
        "target": float(os.environ.get("LENGTH_TARGET", "0.0")),
        "norm": float(os.environ.get("LENGTH_NORM", "512")),
    }


def apply_length_penalty(correctness, response_length, max_length, lam, variant, target_frac):
    if lam == 0.0:
        return float(correctness)

    norm_len = float(response_length) / max(float(max_length), 1.0)  # in [0,1]

    if variant == "plain":
        return float(correctness) - lam * norm_len
    if variant == "conditional": # future work
        penalty = norm_len if correctness > 0 else 0.0
        return float(correctness) - lam * penalty
    if variant == "target": # future work
        return float(correctness) - lam * abs(norm_len - target_frac)

    raise ValueError(f"Unknown LENGTH_VARIANT={variant!r} (plain|conditional|target)")
