from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--time_col", type=str, default=None)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--pclear_col", type=str, required=True)
    p.add_argument("--weather_cols", nargs="+", required=True, help="Training/measured weather columns")
    p.add_argument("--physics_cols", nargs="+", default=[])
    p.add_argument("--eval_weather_cols", nargs="+", default=None, help="Validation/test NWP columns")
    p.add_argument("--eval_physics_cols", nargs="+", default=None)

    p.add_argument("--split_type", choices=["temporal", "station"], default="station")
    p.add_argument("--station_col", type=str, default=None)
    p.add_argument("--train_stations", nargs="*", default=None)
    p.add_argument("--val_stations", nargs="*", default=None)
    p.add_argument("--test_stations", nargs="*", default=None)

    p.add_argument("--settings", nargs="+", default=["direct", "physics_feature", "decomposition"])
    p.add_argument("--backbones", nargs="+", default=["mlp", "tcn", "transformer"])

    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", type=str, default="runs_cross_site_sweep")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_trials_per_pair", type=int, default=None)
    return p.parse_args()


def product_dict(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def get_search_space(backbone: str, setting: str) -> list[dict[str, Any]]:
    if backbone == "mlp":
        grid = {
            "hidden_dim": [64, 128, 256],
            "depth": [2, 3, 4],
            "batch_size": [32, 64, 128],
            "lr": [1e-3, 5e-4],
            "dropout": [0.0, 0.1, 0.2],
        }
    elif backbone == "tcn":
        grid = {
            "hidden_dim": [64, 128, 256],
            "depth": [3, 4, 5],
            "batch_size": [32, 64],
            "lr": [1e-3, 5e-4],
            "dropout": [0.0, 0.1, 0.2],
        }
    elif backbone == "transformer":
        grid = {
            "hidden_dim": [32, 64],
            "depth": [1, 2],
            "batch_size": [64, 128],
            "lr": [5e-4, 2e-4, 1e-4],
            "dropout": [0.1, 0.2, 0.3],
        }
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return product_dict(grid)


def make_run_name(setting: str, backbone: str, cfg: dict[str, Any]) -> str:
    return (
        f"{setting}_{backbone}"
        f"__hd{cfg['hidden_dim']}"
        f"__d{cfg['depth']}"
        f"__bs{cfg['batch_size']}"
        f"__lr{cfg['lr']}"
        f"__do{cfg['dropout']}"
    ).replace(".", "p")


def add_list(cmd: list[str], flag: str, values):
    if values:
        cmd += [flag, *map(str, values)]


def add_if_not_none(cmd: list[str], flag: str, value):
    if value is not None:
        cmd += [flag, str(value)]


def build_cmd(args, setting: str, backbone: str, cfg: dict[str, Any], run_out_root: Path):
    cmd = [
        sys.executable, "train.py",
        "--csv", args.csv,
        "--target_col", args.target_col,
        "--pclear_col", args.pclear_col,
        "--setting", setting,
        "--backbone", backbone,
        "--split_type", args.split_type,
        "--seq_len", str(args.seq_len),
        "--batch_size", str(cfg["batch_size"]),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--lr", str(cfg["lr"]),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--depth", str(cfg["depth"]),
        "--dropout", str(cfg["dropout"]),
        "--device", args.device,
        "--out_root", str(run_out_root),
        "--seed", str(args.seed),
        "--weather_cols", *args.weather_cols,
    ]
    add_list(cmd, "--physics_cols", args.physics_cols)
    add_if_not_none(cmd, "--time_col", args.time_col)
    add_if_not_none(cmd, "--station_col", args.station_col)
    add_list(cmd, "--train_stations", args.train_stations)
    add_list(cmd, "--val_stations", args.val_stations)
    add_list(cmd, "--test_stations", args.test_stations)
    add_list(cmd, "--eval_weather_cols", args.eval_weather_cols)
    add_list(cmd, "--eval_physics_cols", args.eval_physics_cols)
    return cmd


def read_metrics(metrics_path: Path) -> dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_result(setting: str, backbone: str, cfg: dict[str, Any], run_dir: Path, metrics: dict[str, Any]):
    row = {
        "setting": setting,
        "backbone": backbone,
        "run_dir": str(run_dir),
        "hidden_dim": cfg["hidden_dim"],
        "depth": cfg["depth"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "dropout": cfg["dropout"],
        "best_epoch": metrics.get("best_epoch"),
        "best_val_rmse": metrics.get("best_val_rmse"),
        "test_rmse": metrics.get("test", {}).get("rmse"),
        "test_mae": metrics.get("test", {}).get("mae"),
        "test_nrmse": metrics.get("test", {}).get("nrmse"),
        "test_r2": metrics.get("test", {}).get("r2"),
    }
    for station, vals in metrics.get("test_by_station", {}).items():
        row[f"station_{station}_rmse"] = vals.get("rmse")
        row[f"station_{station}_nrmse"] = vals.get("nrmse")
    for subgroup, vals in metrics.get("test_subgroups", {}).items():
        row[f"{subgroup}_rmse"] = vals.get("rmse")
        row[f"{subgroup}_mae"] = vals.get("mae")
        row[f"{subgroup}_n"] = vals.get("n")
    return row


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for setting in args.settings:
        for backbone in args.backbones:
            configs = get_search_space(backbone, setting)
            if args.max_trials_per_pair is not None:
                configs = configs[: args.max_trials_per_pair]

            print("\n" + "=" * 100)
            print(f"[SWEEP] setting={setting}, backbone={backbone}, n_trials={len(configs)}")
            print("=" * 100)

            for trial_idx, cfg in enumerate(configs):
                run_name = make_run_name(setting, backbone, cfg)
                pair_out_root = out_root / run_name
                final_run_dir = pair_out_root / f"{setting}_{backbone}"
                metrics_path = final_run_dir / "metrics.json"

                if args.resume and metrics_path.exists():
                    print(f"[SKIP] {run_name}")
                    metrics = read_metrics(metrics_path)
                    all_rows.append(flatten_result(setting, backbone, cfg, final_run_dir, metrics))
                    continue

                cmd = build_cmd(args, setting, backbone, cfg, pair_out_root)
                print(f"\n[TRIAL {trial_idx + 1}/{len(configs)}] {run_name}")
                print(" ".join(cmd))

                if args.dry_run:
                    continue

                try:
                    subprocess.run(cmd, check=True)
                    metrics = read_metrics(metrics_path)
                    row = flatten_result(setting, backbone, cfg, final_run_dir, metrics)
                    all_rows.append(row)
                    pd.DataFrame(all_rows).to_csv(out_root / "sweep_results_running.csv", index=False)
                    print(f"[DONE] val_rmse={row['best_val_rmse']:.4f}, test_rmse={row['test_rmse']:.4f}")
                except subprocess.CalledProcessError as e:
                    print(f"[FAILED] {run_name}: {e}")
                    all_rows.append({**cfg, "setting": setting, "backbone": backbone, "run_dir": str(final_run_dir), "failed": True})
                    pd.DataFrame(all_rows).to_csv(out_root / "sweep_results_running.csv", index=False)

    if args.dry_run:
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_root / "sweep_results.csv", index=False)

    if not df.empty and "best_val_rmse" in df.columns:
        valid = df.dropna(subset=["best_val_rmse"]).copy()
        valid = valid.sort_values(["setting", "backbone", "best_val_rmse"])
        best = valid.groupby(["setting", "backbone"], as_index=False).first().sort_values(["backbone", "setting"])
        best.to_csv(out_root / "best_by_setting_backbone.csv", index=False)
        cols = ["setting", "backbone", "hidden_dim", "depth", "batch_size", "lr", "dropout", "best_val_rmse", "test_rmse"]
        extra = [c for c in ["cloudy_rmse", "peak_rmse", "ramp_rmse"] if c in best.columns]
        print("\n[BEST BY SETTING/BACKBONE]")
        print(best[cols + extra].to_string(index=False))

    print(f"\n[SAVED] {out_root / 'sweep_results.csv'}")
    print(f"[SAVED] {out_root / 'best_by_setting_backbone.csv'}")


if __name__ == "__main__":
    main()
