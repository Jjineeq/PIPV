from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--time_col", type=str, default=None)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--pclear_col", type=str, required=True)
    p.add_argument("--weather_cols", nargs="+", required=True)
    p.add_argument("--physics_cols", nargs="+", default=[])
    p.add_argument("--eval_weather_cols", nargs="+", default=None)
    p.add_argument("--eval_physics_cols", nargs="+", default=None)

    p.add_argument("--split_type", choices=["temporal", "station"], default="station")
    p.add_argument("--station_col", type=str, default=None)
    p.add_argument("--capacity_col", type=str, default="capacity")
    p.add_argument("--train_stations", nargs="*", default=None)
    p.add_argument("--val_stations", nargs="*", default=None)
    p.add_argument("--test_stations", nargs="*", default=None)

    p.add_argument("--settings", nargs="+", default=["direct", "physics_feature", "decomposition", "clear_sky_film"])
    p.add_argument("--backbones", nargs="+", default=["mlp", "tcn", "transformer"])
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--residual_scale", type=float, default=0.2)
    p.add_argument("--residual_penalty", type=float, default=0.01)
    p.add_argument("--film_modulation_scale", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_root", type=str, default="runs_cross_site")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def add_if_not_none(cmd: list[str], flag: str, value):
    if value is not None:
        cmd += [flag, str(value)]


def add_list(cmd: list[str], flag: str, values):
    if values:
        cmd += [flag, *map(str, values)]


def main():
    args = parse_args()

    for setting in args.settings:
        for backbone in args.backbones:
            cmd = [
                sys.executable, "train.py",
                "--csv", args.csv,
                "--target_col", args.target_col,
                "--pclear_col", args.pclear_col,
                "--setting", setting,
                "--backbone", backbone,
                "--split_type", args.split_type,
                "--seq_len", str(args.seq_len),
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--lr", str(args.lr),
                "--hidden_dim", str(args.hidden_dim),
                "--depth", str(args.depth),
                "--dropout", str(args.dropout),
                "--residual_scale", str(args.residual_scale),
                "--residual_penalty", str(args.residual_penalty),
                "--film_modulation_scale", str(args.film_modulation_scale),
                "--device", args.device,
                "--out_root", args.out_root,
                "--seed", str(args.seed),
                "--weather_cols", *args.weather_cols,
            ]
            add_list(cmd, "--physics_cols", args.physics_cols)
            add_if_not_none(cmd, "--time_col", args.time_col)
            add_if_not_none(cmd, "--station_col", args.station_col)
            add_if_not_none(cmd, "--capacity_col", args.capacity_col)
            add_list(cmd, "--train_stations", args.train_stations)
            add_list(cmd, "--val_stations", args.val_stations)
            add_list(cmd, "--test_stations", args.test_stations)
            add_list(cmd, "--eval_weather_cols", args.eval_weather_cols)
            add_list(cmd, "--eval_physics_cols", args.eval_physics_cols)

            print("\n[RUN]", setting, backbone)
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
