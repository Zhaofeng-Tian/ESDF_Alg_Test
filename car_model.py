# car_model.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class DiffDriveState:
    x: float
    y: float
    theta: float  # radians

def wrap_angle(theta: float) -> float:
    return float((theta + np.pi) % (2 * np.pi) - np.pi)

def headings_from_path(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    For diff-drive, you can set theta as tangent direction of the path.
    """
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    th = np.arctan2(dy, dx)
    return th
