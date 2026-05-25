from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def regression_metrics(y_true, y_pred, capacity=None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "nrmse": np.nan, "r2": np.nan}
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(capacity) if capacity is not None else float(np.nanpercentile(y_true, 99))
    denom = max(denom, 1e-8)
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan
    return {"rmse": rmse, "mae": mae, "nrmse": rmse / denom, "r2": r2}


def build_eval_masks(df_pred: pd.DataFrame, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    y = df_pred["y_true"].to_numpy(dtype=float)
    p = df_pred["p_clear"].to_numpy(dtype=float)

    day = p > max(np.nanpercentile(p, 5), eps)
    ratio = np.clip(y / (p + eps), 0.0, 2.0)

    p_day = p[day]
    peak_thr = np.nanpercentile(p_day, 80) if len(p_day) > 0 else np.nanpercentile(p, 80)

    cloudy = day & (ratio < 0.6)
    clear_like = day & (ratio >= 0.75)
    peak = day & (p >= peak_thr)

    dy = np.zeros_like(y)
    dy[1:] = np.abs(np.diff(y))
    ramp_thr = np.nanpercentile(dy[day], 80) if np.any(day) else np.nanpercentile(dy, 80)
    ramp = day & (dy >= ramp_thr)

    return {
        "all": np.ones_like(y, dtype=bool),
        "daytime": day,
        "clear_like": clear_like,
        "cloudy": cloudy,
        "peak": peak,
        "ramp": ramp,
    }


def subgroup_metrics(df_pred: pd.DataFrame, capacity=None) -> Dict[str, Dict[str, float]]:
    masks = build_eval_masks(df_pred)
    out = {}
    for name, mask in masks.items():
        sub = df_pred.loc[mask]
        out[name] = regression_metrics(sub["y_true"], sub["y_pred"], capacity=capacity)
        out[name]["n"] = int(mask.sum())
    return out
