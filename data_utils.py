from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset


@dataclass
class SplitData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scalers: Dict[str, MinMaxScaler]
    meta: Dict


class PVDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        p_clear: np.ndarray,
        seq_len: int = 24,
        X_phys: Optional[np.ndarray] = None,
        capacity: Optional[np.ndarray] = None,
        station_id: Optional[np.ndarray] = None,
        timestamp: Optional[Sequence] = None,
    ):
        assert len(X) == len(y) == len(p_clear)
        if X_phys is not None:
            assert len(X_phys) == len(X)
        self.X = X.astype(np.float32)
        self.X_phys = None if X_phys is None else X_phys.astype(np.float32)
        self.y = y.astype(np.float32)
        self.p_clear = p_clear.astype(np.float32)
        self.capacity = np.ones(len(y), dtype=np.float32) if capacity is None else capacity.astype(np.float32)
        self.station_id = np.full(len(y), -1, dtype=np.int64) if station_id is None else station_id.astype(np.int64)
        self.timestamp = np.array(["" for _ in range(len(y))], dtype=object) if timestamp is None else np.asarray(timestamp, dtype=object)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        j = idx + self.seq_len - 1
        out: Dict[str, torch.Tensor | str] = {
            "x": torch.from_numpy(self.X[idx : j + 1]),
            "y": torch.tensor(self.y[j], dtype=torch.float32),
            "p_clear": torch.tensor(self.p_clear[j], dtype=torch.float32),
            "capacity": torch.tensor(self.capacity[j], dtype=torch.float32),
            "station_id": torch.tensor(self.station_id[j], dtype=torch.long),
            "timestamp": str(self.timestamp[j]),
        }
        if self.X_phys is not None:
            out["x_phys"] = torch.from_numpy(self.X_phys[idx : j + 1])
        return out


def add_time_features(df: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    if time_col is not None and time_col in df.columns:
        dt = pd.to_datetime(df[time_col])
    else:
        dt = pd.to_datetime(df.index)

    hour = dt.hour.to_numpy() if hasattr(dt, "hour") else dt.dt.hour.to_numpy()
    doy = dt.dayofyear.to_numpy() if hasattr(dt, "dayofyear") else dt.dt.dayofyear.to_numpy()

    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    if "hour_cos" not in df.columns:
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    if "doy_sin" not in df.columns:
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    if "doy_cos" not in df.columns:
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def load_dataframe(csv_path: str, time_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if time_col is not None and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        first = df.columns[0]
        if first.lower().startswith("unnamed"):
            df = df.drop(columns=[first])

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both").ffill().bfill()
    return df


def temporal_split_indices(n: int, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[slice, slice, slice]:
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    return slice(0, i_train), slice(i_train, i_val), slice(i_val, n)


def _fit_scaler(arr: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(arr)
    return scaler


def _to_station_set(values) -> Optional[set[int]]:
    if values is None:
        return None
    if len(values) == 0:
        return None
    return {int(v) for v in values}


def _timestamps(df: pd.DataFrame, time_col: Optional[str]) -> np.ndarray:
    if time_col is not None and time_col in df.columns:
        return pd.to_datetime(df[time_col]).astype(str).to_numpy()
    return pd.to_datetime(df.index).astype(str).to_numpy()


def _make_concat_dataset(
    df_split: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    pclear_col: str,
    seq_len: int,
    x_scaler: MinMaxScaler,
    time_col: Optional[str],
    station_col: Optional[str],
    capacity_col: Optional[str],
    physics_cols: Optional[List[str]] = None,
    phys_scaler: Optional[MinMaxScaler] = None,
) -> Dataset:
    datasets = []
    group_iter = [(None, df_split)] if station_col is None else df_split.groupby(station_col, sort=True)

    for _, g in group_iter:
        g = g.sort_values(time_col) if time_col is not None and time_col in g.columns else g.sort_index()
        if len(g) < seq_len:
            continue

        X = x_scaler.transform(g[feature_cols].to_numpy(dtype=float))
        X_phys = None
        if physics_cols:
            if phys_scaler is None:
                raise ValueError("phys_scaler is required when physics_cols are provided")
            X_phys = phys_scaler.transform(g[physics_cols].to_numpy(dtype=float))

        y = g[target_col].to_numpy(dtype=float)
        p = g[pclear_col].to_numpy(dtype=float)
        cap = g[capacity_col].to_numpy(dtype=float) if capacity_col is not None and capacity_col in g.columns else np.ones(len(g), dtype=float)
        sid = g[station_col].to_numpy(dtype=int) if station_col is not None and station_col in g.columns else np.full(len(g), -1, dtype=int)
        ts = _timestamps(g, time_col)

        datasets.append(PVDataset(X, y, p, seq_len, X_phys=X_phys, capacity=cap, station_id=sid, timestamp=ts))

    if not datasets:
        raise ValueError("No valid windows were created. Check split/station IDs and seq_len.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_dataloaders(
    df: pd.DataFrame,
    target_col: str,
    pclear_col: str,
    feature_cols: List[str],
    seq_len: int = 24,
    batch_size: int = 64,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    eval_feature_cols: Optional[List[str]] = None,
    physics_cols: Optional[List[str]] = None,
    eval_physics_cols: Optional[List[str]] = None,
    split_type: str = "temporal",
    station_col: Optional[str] = None,
    train_stations: Optional[Sequence[int | str]] = None,
    val_stations: Optional[Sequence[int | str]] = None,
    test_stations: Optional[Sequence[int | str]] = None,
    capacity_col: Optional[str] = "capacity",
    time_col: Optional[str] = None,
) -> SplitData:
    eval_feature_cols = eval_feature_cols or feature_cols
    physics_cols = physics_cols or []
    eval_physics_cols = eval_physics_cols or physics_cols

    if len(eval_feature_cols) != len(feature_cols):
        raise ValueError("eval_feature_cols must have the same length/order semantics as feature_cols")
    if len(eval_physics_cols) != len(physics_cols):
        raise ValueError("eval_physics_cols must have the same length/order semantics as physics_cols")

    required = [target_col, pclear_col] + feature_cols + eval_feature_cols + physics_cols + eval_physics_cols
    if station_col is not None:
        required.append(station_col)
    if capacity_col is not None:
        required.append(capacity_col)
    if time_col is not None:
        required.append(time_col)
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=list(dict.fromkeys(required))).copy()

    if split_type == "station":
        if station_col is None:
            raise ValueError("station_col is required when split_type='station'")
        tr_set, va_set, te_set = map(_to_station_set, [train_stations, val_stations, test_stations])
        if tr_set is None or va_set is None or te_set is None:
            raise ValueError("train_stations, val_stations, and test_stations are required for station split")
        df_train = df[df[station_col].astype(int).isin(tr_set)].copy()
        df_val = df[df[station_col].astype(int).isin(va_set)].copy()
        df_test = df[df[station_col].astype(int).isin(te_set)].copy()
    elif split_type == "temporal":
        df = df.sort_values(time_col) if time_col is not None and time_col in df.columns else df.sort_index()
        s_train, s_val, s_test = temporal_split_indices(len(df), train_ratio, val_ratio)
        df_train, df_val, df_test = df.iloc[s_train].copy(), df.iloc[s_val].copy(), df.iloc[s_test].copy()
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    x_scaler = _fit_scaler(df_train[feature_cols].to_numpy(dtype=float))
    phys_scaler = None
    if physics_cols:
        phys_scaler = _fit_scaler(df_train[physics_cols].to_numpy(dtype=float))

    train_ds = _make_concat_dataset(
        df_train, feature_cols, target_col, pclear_col, seq_len, x_scaler, time_col, station_col, capacity_col,
        physics_cols=physics_cols, phys_scaler=phys_scaler,
    )
    val_ds = _make_concat_dataset(
        df_val, eval_feature_cols, target_col, pclear_col, seq_len, x_scaler, time_col, station_col, capacity_col,
        physics_cols=eval_physics_cols, phys_scaler=phys_scaler,
    )
    test_ds = _make_concat_dataset(
        df_test, eval_feature_cols, target_col, pclear_col, seq_len, x_scaler, time_col, station_col, capacity_col,
        physics_cols=eval_physics_cols, phys_scaler=phys_scaler,
    )

    return SplitData(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        scalers={"x": x_scaler, "physics": phys_scaler},
        meta={
            "n_total": int(len(df)),
            "n_train_rows": int(len(df_train)),
            "n_val_rows": int(len(df_val)),
            "n_test_rows": int(len(df_test)),
            "n_train_windows": len(train_ds),
            "n_val_windows": len(val_ds),
            "n_test_windows": len(test_ds),
            "feature_cols": feature_cols,
            "eval_feature_cols": eval_feature_cols,
            "physics_cols": physics_cols,
            "eval_physics_cols": eval_physics_cols,
            "target_col": target_col,
            "pclear_col": pclear_col,
            "capacity_col": capacity_col,
            "station_col": station_col,
            "split_type": split_type,
            "train_stations": sorted(list(_to_station_set(train_stations) or [])),
            "val_stations": sorted(list(_to_station_set(val_stations) or [])),
            "test_stations": sorted(list(_to_station_set(test_stations) or [])),
            "seq_len": seq_len,
        },
    )
