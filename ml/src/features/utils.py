import numpy as np


def euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance in normalized coordinates (x, y only)."""
    return float(np.linalg.norm(p1[:2] - p2[:2]))