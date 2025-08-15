from __future__ import annotations

import math


def sigmoid(z: float, eps=1e-12) -> float:
    if z < -700:
        return eps
    if z > 700:
        return 1 - eps

    s = 1 / (1 + math.exp(-z))
    s = min(max(s, eps), 1 - eps)
    return s
