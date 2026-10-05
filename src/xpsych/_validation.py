"""Shared numerical input checks; no model or data access."""
import numpy as np


def matrix(value, name, *, min_rows=1, min_cols=1):
    out = np.asarray(value, dtype=float)
    if out.ndim != 2 or out.shape[0] < min_rows or out.shape[1] < min_cols:
        raise ValueError(f"{name} must be a 2-D array with at least {min_rows} rows and {min_cols} columns")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def square(value, name, *, min_size=2, psd=False):
    out = matrix(value, name, min_rows=min_size, min_cols=min_size)
    if out.shape[0] != out.shape[1] or not np.allclose(out, out.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be square and symmetric")
    if psd and np.linalg.eigvalsh(out).min() < -1e-10 * max(1, np.linalg.norm(out, 2)):
        raise ValueError(f"{name} must be positive semidefinite")
    return out


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def variable(value, name):
    if np.ptp(value) <= np.finfo(float).eps * max(1, np.max(np.abs(value))):
        raise ValueError(f"{name} is constant; correlation is undefined")


def labels(value, n, name):
    if len(value) != n or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{name} must contain one nonempty string per row")
    return np.asarray(value)
