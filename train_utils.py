from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from metrics import regression_metrics, subgroup_metrics


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _forward_batch(model, batch, device):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    p_clear = batch["p_clear"].to(device)
    x_phys = batch.get("x_phys")
    if x_phys is not None:
        x_phys = x_phys.to(device)
    out = model(x, p_clear=p_clear, x_phys=x_phys)
    return out, y, p_clear


def train_one_epoch(model, loader, optimizer, device, residual_penalty: float = 0.01):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    n = 0

    for batch in loader:
        out, y, _ = _forward_batch(model, batch, device)
        y_hat = out["y_hat"]

        loss = mse(y_hat, y)

        if "r_hat" in out and residual_penalty > 0:
            loss = loss + residual_penalty * torch.mean(out["r_hat"] ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        out, y, p_clear = _forward_batch(model, batch, device)
        y_hat = out["y_hat"]
        c_hat = out.get("c_hat", torch.full_like(y_hat, float("nan")))
        r_hat = out.get("r_hat", torch.full_like(y_hat, float("nan")))
        gamma_mean = out.get("film_gamma_mean", torch.full_like(y_hat, float("nan")))
        beta_mean = out.get("film_beta_mean", torch.full_like(y_hat, float("nan")))

        capacity = batch.get("capacity", torch.ones_like(y.cpu())).detach().cpu().numpy().astype(float)
        station_id = batch.get("station_id", torch.full_like(y.cpu().long(), -1)).detach().cpu().numpy().astype(int)
        timestamps = batch.get("timestamp", [""] * len(y))

        y_np = y.detach().cpu().numpy().astype(float)
        yh_np = y_hat.detach().cpu().numpy().astype(float)
        pc_np = p_clear.detach().cpu().numpy().astype(float)

        for i in range(len(y_np)):
            rows.append({
                "timestamp": str(timestamps[i]),
                "station_id": int(station_id[i]),
                "capacity": float(capacity[i]),
                "y_true": float(y_np[i]),
                "y_pred": float(yh_np[i]),
                "p_clear": float(pc_np[i]),
                "c_hat": float(c_hat[i].detach().cpu()),
                "r_hat": float(r_hat[i].detach().cpu()),
                "film_gamma_mean": float(gamma_mean[i].detach().cpu()),
                "film_beta_mean": float(beta_mean[i].detach().cpu()),
                "y_true_power": float(y_np[i] * capacity[i]),
                "y_pred_power": float(yh_np[i] * capacity[i]),
                "p_clear_power": float(pc_np[i] * capacity[i]),
            })
    return pd.DataFrame(rows)


def _by_station_metrics(df_pred: pd.DataFrame, y_true_col: str, y_pred_col: str):
    out = {}
    if "station_id" not in df_pred.columns:
        return out
    for sid, g in df_pred.groupby("station_id"):
        m = regression_metrics(g[y_true_col], g[y_pred_col])
        m["n"] = int(len(g))
        out[str(int(sid))] = m
    return out


def fit_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    out_dir,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    patience=20,
    device="cuda",
    residual_penalty: float = 0.01,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    history = []

    pbar = tqdm(range(1, epochs + 1), desc=f"train:{out_dir.name}")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, device, residual_penalty=residual_penalty)

        val_pred = predict(model, val_loader, device)
        val_metrics = regression_metrics(val_pred["y_true"], val_pred["y_pred"])
        val_rmse = val_metrics["rmse"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse_norm": val_rmse,
            "val_mae_norm": val_metrics["mae"],
            "val_nrmse_norm": val_metrics["nrmse"],
            "val_r2_norm": val_metrics["r2"],
        })

        pbar.set_postfix({"train": f"{train_loss:.6f}", "val_rmse": f"{val_rmse:.6f}", "best": f"{best_val:.6f}"})

        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            bad += 1

        if bad >= patience:
            break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    val_pred = predict(model, val_loader, device)
    test_pred = predict(model, test_loader, device)
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
    test_pred.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_rmse_norm": best_val,
        "val": {
            "normalized": regression_metrics(val_pred["y_true"], val_pred["y_pred"]),
            "power": regression_metrics(val_pred["y_true_power"], val_pred["y_pred_power"]),
            "normalized_by_station": _by_station_metrics(val_pred, "y_true", "y_pred"),
            "power_by_station": _by_station_metrics(val_pred, "y_true_power", "y_pred_power"),
            "normalized_subgroups": subgroup_metrics(val_pred),
        },
        "test": {
            "normalized": regression_metrics(test_pred["y_true"], test_pred["y_pred"]),
            "power": regression_metrics(test_pred["y_true_power"], test_pred["y_pred_power"]),
            "normalized_by_station": _by_station_metrics(test_pred, "y_true", "y_pred"),
            "power_by_station": _by_station_metrics(test_pred, "y_true_power", "y_pred_power"),
            "normalized_subgroups": subgroup_metrics(test_pred),
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
