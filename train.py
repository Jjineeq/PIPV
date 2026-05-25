from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import load_dataframe, add_time_features, build_dataloaders
from models import build_model
from train_utils import set_seed, fit_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--time_col", type=str, default=None)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--pclear_col", type=str, required=True)
    p.add_argument("--weather_cols", nargs="+", required=True)
    p.add_argument("--eval_weather_cols", nargs="+", default=None)
    p.add_argument("--physics_cols", nargs="+", default=[])
    p.add_argument("--eval_physics_cols", nargs="+", default=None)

    p.add_argument("--setting", choices=["direct", "physics_feature", "decomposition", "clear_sky_film"], required=True)
    p.add_argument("--backbone", choices=["mlp", "tcn", "transformer"], required=True)

    p.add_argument("--split_type", choices=["temporal", "station"], default="temporal")
    p.add_argument("--station_col", type=str, default=None)
    p.add_argument("--capacity_col", type=str, default="capacity")
    p.add_argument("--train_stations", nargs="*", default=None)
    p.add_argument("--val_stations", nargs="*", default=None)
    p.add_argument("--test_stations", nargs="*", default=None)

    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--c_max", type=float, default=1.2)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--residual_penalty", type=float, default=0.01)
    p.add_argument("--film_modulation_scale", type=float, default=0.1)

    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", type=str, default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    df = load_dataframe(args.csv, time_col=args.time_col)
    df = add_time_features(df, time_col=args.time_col)

    eval_weather_cols = args.eval_weather_cols or args.weather_cols
    eval_physics_cols = args.eval_physics_cols or args.physics_cols

    if args.setting == "physics_feature":
        feature_cols = list(dict.fromkeys(args.weather_cols + args.physics_cols))
        eval_feature_cols = list(dict.fromkeys(eval_weather_cols + eval_physics_cols))
        film_physics_cols = []
        eval_film_physics_cols = []
    else:
        feature_cols = args.weather_cols
        eval_feature_cols = eval_weather_cols
        film_physics_cols = args.physics_cols if args.setting == "clear_sky_film" else []
        eval_film_physics_cols = eval_physics_cols if args.setting == "clear_sky_film" else []

    split = build_dataloaders(
        df=df,
        target_col=args.target_col,
        pclear_col=args.pclear_col,
        feature_cols=feature_cols,
        eval_feature_cols=eval_feature_cols,
        physics_cols=film_physics_cols,
        eval_physics_cols=eval_film_physics_cols,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        split_type=args.split_type,
        station_col=args.station_col,
        train_stations=args.train_stations,
        val_stations=args.val_stations,
        test_stations=args.test_stations,
        capacity_col=args.capacity_col,
        time_col=args.time_col,
    )

    model = build_model(
        setting=args.setting,
        backbone_name=args.backbone,
        input_dim=len(feature_cols),
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        c_max=args.c_max,
        residual_scale=args.residual_scale,
        physics_input_dim=len(film_physics_cols),
        film_modulation_scale=args.film_modulation_scale,
    )

    out_dir = Path(args.out_root) / f"{args.setting}_{args.backbone}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["feature_cols_used"] = feature_cols
    config["eval_feature_cols_used"] = eval_feature_cols
    config["film_physics_cols_used"] = film_physics_cols
    config["eval_film_physics_cols_used"] = eval_film_physics_cols
    config["data_meta"] = split.meta
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    metrics = fit_model(
        model=model,
        train_loader=split.train_loader,
        val_loader=split.val_loader,
        test_loader=split.test_loader,
        out_dir=out_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        residual_penalty=args.residual_penalty,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
