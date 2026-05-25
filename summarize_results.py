from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_setting_backbone(run_name: str):
    known_backbones = ["transformer", "tcn", "mlp"]
    for b in known_backbones:
        suffix = f"_{b}"
        if run_name.endswith(suffix):
            return run_name[: -len(suffix)], b
    parts = run_name.split("_")
    return "_".join(parts[:-1]), parts[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs_pvod_film")
    p.add_argument("--out_csv", type=str, default="runs_pvod_film/summary_metrics.csv")
    args = p.parse_args()

    rows = []

    for mpath in sorted(Path(args.runs_dir).glob("*/metrics.json")):
        run_name = mpath.parent.name

        with open(mpath, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        setting, backbone = infer_setting_backbone(run_name)
        test_norm = get_nested(metrics, ["test", "normalized"], {})
        val_norm = get_nested(metrics, ["val", "normalized"], {})
        test_power = get_nested(metrics, ["test", "power"], {})

        row = {
            "run": run_name,
            "setting": setting,
            "backbone": backbone,
            "best_epoch": metrics.get("best_epoch"),
            "best_val_rmse_norm": metrics.get("best_val_rmse_norm", val_norm.get("rmse")),
            "val_nrmse_norm": val_norm.get("nrmse"),
            "val_r2_norm": val_norm.get("r2"),
            "test_rmse_norm": test_norm.get("rmse"),
            "test_mae_norm": test_norm.get("mae"),
            "test_nrmse_norm": test_norm.get("nrmse"),
            "test_r2_norm": test_norm.get("r2"),
            "test_rmse_power": test_power.get("rmse"),
            "test_mae_power": test_power.get("mae"),
            "test_nrmse_power": test_power.get("nrmse"),
            "test_r2_power": test_power.get("r2"),
        }

        for sid, vals in get_nested(metrics, ["test", "normalized_by_station"], {}).items():
            row[f"station{sid}_rmse_norm"] = vals.get("rmse")
            row[f"station{sid}_mae_norm"] = vals.get("mae")
            row[f"station{sid}_nrmse_norm"] = vals.get("nrmse")
            row[f"station{sid}_r2_norm"] = vals.get("r2")
            row[f"station{sid}_n"] = vals.get("n")

        for sid, vals in get_nested(metrics, ["test", "power_by_station"], {}).items():
            row[f"station{sid}_rmse_power"] = vals.get("rmse")
            row[f"station{sid}_mae_power"] = vals.get("mae")
            row[f"station{sid}_nrmse_power"] = vals.get("nrmse")
            row[f"station{sid}_r2_power"] = vals.get("r2")

        for subgroup, vals in get_nested(metrics, ["test", "normalized_subgroups"], {}).items():
            row[f"{subgroup}_rmse_norm"] = vals.get("rmse")
            row[f"{subgroup}_mae_norm"] = vals.get("mae")
            row[f"{subgroup}_nrmse_norm"] = vals.get("nrmse")
            row[f"{subgroup}_r2_norm"] = vals.get("r2")
            row[f"{subgroup}_n"] = vals.get("n")

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["test_nrmse_norm", "test_rmse_norm"], na_position="last")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    show_cols = [
        "run", "setting", "backbone", "best_epoch",
        "best_val_rmse_norm", "test_rmse_norm", "test_mae_norm", "test_nrmse_norm", "test_r2_norm",
        "station7_nrmse_norm", "station8_nrmse_norm", "station9_nrmse_norm",
        "daytime_nrmse_norm", "cloudy_nrmse_norm", "peak_nrmse_norm", "ramp_nrmse_norm",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    print(df[show_cols].to_string(index=False))
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
