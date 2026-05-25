#!/usr/bin/env bash
set -euo pipefail

# Sequential 5-seed Leave-One-Station-Out evaluation for PVOD
# - test: one station k
# - validation: next two stations (k+1)%10, (k+2)%10
# - train: remaining seven stations
# - settings: direct, physics_feature, decomposition only; clear_sky_film excluded
# - seeds run sequentially, not in parallel
#
# Outputs per seed:
#   runs_pvod_loso_val2_seed{seed}/foldXX_testY/<setting>_<backbone>/metrics.json
#   runs_pvod_loso_val2_seed{seed}/loso_fold_results.csv
#   runs_pvod_loso_val2_seed{seed}/loso_summary_mean_std.csv
#
# Combined outputs:
#   runs_pvod_loso_val2_seed5/loso_all_seeds_fold_results.csv
#   runs_pvod_loso_val2_seed5/loso_all_seeds_summary_mean_std.csv

CSV=${CSV:-datasets/PVOD_processed.csv}
BASE_OUT_ROOT=${BASE_OUT_ROOT:-runs_pvod_loso_val2_seed5}
LOG_DIR=${LOG_DIR:-logs}
DEVICE=${DEVICE:-cuda}
SEEDS_STR=${SEEDS:-"1 2 3 4 5"}

# Hyperparameters
SEQ_LEN=${SEQ_LEN:-24}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-200}
PATIENCE=${PATIENCE:-20}
LR=${LR:-5e-4}
HIDDEN_DIM=${HIDDEN_DIM:-128}
DEPTH=${DEPTH:-4}
DROPOUT=${DROPOUT:-0.1}
RESIDUAL_SCALE=${RESIDUAL_SCALE:-0.2}

# Methods/backbones. FiLM intentionally excluded.
SETTINGS_STR=${SETTINGS:-"direct physics_feature decomposition"}
BACKBONES_STR=${BACKBONES:-"mlp tcn transformer"}

mkdir -p "${BASE_OUT_ROOT}" "${LOG_DIR}"

run_one_seed() {
  local SEED_VALUE="$1"
  local OUT_ROOT="${BASE_OUT_ROOT}/seed${SEED_VALUE}"
  mkdir -p "${OUT_ROOT}"

  echo ""
  echo "################################################################################"
  echo "[SEED START] seed=${SEED_VALUE}"
  echo "[OUT_ROOT] ${OUT_ROOT}"
  echo "################################################################################"

  for TEST in 0 1 2 3 4 5 6 7 8 9; do
    local VAL1=$(( (TEST + 1) % 10 ))
    local VAL2=$(( (TEST + 2) % 10 ))

    TRAIN_STATIONS=()
    for S in 0 1 2 3 4 5 6 7 8 9; do
      if [[ "${S}" != "${TEST}" && "${S}" != "${VAL1}" && "${S}" != "${VAL2}" ]]; then
        TRAIN_STATIONS+=("${S}")
      fi
    done

    local FOLD_DIR="${OUT_ROOT}/fold$(printf '%02d' ${TEST})_test${TEST}"

    echo ""
    echo "================================================================================"
    echo "[FOLD] seed=${SEED_VALUE} | test=${TEST} | val=${VAL1} ${VAL2} | train=${TRAIN_STATIONS[*]}"
    echo "[FOLD_DIR] ${FOLD_DIR}"
    echo "================================================================================"

    python run_all.py \
      --csv "${CSV}" \
      --time_col datetime \
      --target_col target_power_norm \
      --pclear_col P_clear_norm \
      --split_type station \
      --station_col station_id \
      --train_stations "${TRAIN_STATIONS[@]}" \
      --val_stations "${VAL1}" "${VAL2}" \
      --test_stations "${TEST}" \
      --settings ${SETTINGS_STR} \
      --backbones ${BACKBONES_STR} \
      --weather_cols \
        meas_globalirrad \
        meas_temperature \
        meas_pressure \
        meas_winddirection \
        meas_windspeed \
        hour_sin \
        hour_cos \
        doy_sin \
        doy_cos \
      --eval_weather_cols \
        nwp_globalirrad \
        nwp_temperature \
        nwp_pressure \
        nwp_winddirection \
        nwp_windspeed \
        hour_sin \
        hour_cos \
        doy_sin \
        doy_cos \
      --physics_cols \
        P_clear_norm \
        longitude \
        latitude \
        array_tilt \
        array_azimuth \
        hour_sin \
        hour_cos \
        doy_sin \
        doy_cos \
      --seq_len "${SEQ_LEN}" \
      --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --lr "${LR}" \
      --hidden_dim "${HIDDEN_DIM}" \
      --depth "${DEPTH}" \
      --dropout "${DROPOUT}" \
      --residual_scale "${RESIDUAL_SCALE}" \
      --device "${DEVICE}" \
      --out_root "${FOLD_DIR}" \
      --seed "${SEED_VALUE}"
  done

  summarize_one_seed "${OUT_ROOT}" "${SEED_VALUE}"
  echo "[SEED DONE] seed=${SEED_VALUE}"
}

summarize_one_seed() {
  local OUT_ROOT="$1"
  local SEED_VALUE="$2"

  OUT_ROOT="${OUT_ROOT}" SEED_VALUE="${SEED_VALUE}" python - <<'PY'
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

OUT_ROOT = Path(os.environ["OUT_ROOT"])
SEED_VALUE = int(os.environ["SEED_VALUE"])


def get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_setting_backbone(run_name: str):
    for b in ["transformer", "tcn", "mlp"]:
        suffix = f"_{b}"
        if run_name.endswith(suffix):
            return run_name[:-len(suffix)], b
    parts = run_name.split("_")
    return "_".join(parts[:-1]), parts[-1]

rows = []
for fold_dir in sorted(OUT_ROOT.glob("fold*_test*")):
    m = re.search(r"test(\d+)$", fold_dir.name)
    test_station = int(m.group(1)) if m else None

    for metrics_path in sorted(fold_dir.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        setting, backbone = infer_setting_backbone(run_name)

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        val_norm = get_nested(metrics, ["val", "normalized"], {}) or {}
        test_norm = get_nested(metrics, ["test", "normalized"], {}) or {}
        station_metrics = get_nested(metrics, ["test", "normalized_by_station", str(test_station)], {}) or {}
        subgroups = get_nested(metrics, ["test", "normalized_subgroups"], {}) or {}

        row = {
            "seed": SEED_VALUE,
            "fold": fold_dir.name,
            "test_station": test_station,
            "run": run_name,
            "setting": setting,
            "backbone": backbone,
            "best_epoch": metrics.get("best_epoch"),
            "best_val_rmse_norm": metrics.get("best_val_rmse_norm", val_norm.get("rmse")),
            "val_rmse_norm": val_norm.get("rmse"),
            "val_mae_norm": val_norm.get("mae"),
            "val_nrmse_norm": val_norm.get("nrmse"),
            "val_r2_norm": val_norm.get("r2"),
            "test_rmse_norm": test_norm.get("rmse"),
            "test_mae_norm": test_norm.get("mae"),
            "test_nrmse_norm": test_norm.get("nrmse"),
            "test_r2_norm": test_norm.get("r2"),
            "test_station_rmse_norm": station_metrics.get("rmse", test_norm.get("rmse")),
            "test_station_mae_norm": station_metrics.get("mae", test_norm.get("mae")),
            "test_station_nrmse_norm": station_metrics.get("nrmse", test_norm.get("nrmse")),
            "test_station_r2_norm": station_metrics.get("r2", test_norm.get("r2")),
        }

        for name in ["daytime", "clear_like", "cloudy", "peak", "ramp"]:
            vals = subgroups.get(name, {})
            row[f"{name}_rmse_norm"] = vals.get("rmse")
            row[f"{name}_mae_norm"] = vals.get("mae")
            row[f"{name}_nrmse_norm"] = vals.get("nrmse")
            row[f"{name}_r2_norm"] = vals.get("r2")
            row[f"{name}_n"] = vals.get("n")

        rows.append(row)

if not rows:
    raise RuntimeError(f"No metrics.json found under {OUT_ROOT}")

fold_df = pd.DataFrame(rows)
fold_csv = OUT_ROOT / "loso_fold_results.csv"
fold_df.to_csv(fold_csv, index=False)

metric_cols = [
    "test_rmse_norm", "test_mae_norm", "test_nrmse_norm", "test_r2_norm",
    "test_station_rmse_norm", "test_station_mae_norm", "test_station_nrmse_norm", "test_station_r2_norm",
    "daytime_nrmse_norm", "cloudy_nrmse_norm", "peak_nrmse_norm", "ramp_nrmse_norm",
]
metric_cols = [c for c in metric_cols if c in fold_df.columns]

summary_rows = []
for (setting, backbone), g in fold_df.groupby(["setting", "backbone"]):
    row = {"seed": SEED_VALUE, "setting": setting, "backbone": backbone, "n_folds": len(g)}
    for c in metric_cols:
        vals = pd.to_numeric(g[c], errors="coerce")
        row[f"{c}_mean"] = vals.mean()
        row[f"{c}_std"] = vals.std(ddof=1)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty and "test_nrmse_norm_mean" in summary_df.columns:
    summary_df = summary_df.sort_values("test_nrmse_norm_mean")

summary_csv = OUT_ROOT / "loso_summary_mean_std.csv"
summary_df.to_csv(summary_csv, index=False)

print("\n[SEED FOLD RESULTS]")
show_fold_cols = ["seed", "test_station", "setting", "backbone", "test_nrmse_norm", "test_mae_norm", "test_r2_norm"]
print(fold_df[show_fold_cols].sort_values(["test_station", "test_nrmse_norm"]).to_string(index=False))

print("\n[SEED SUMMARY: mean ± std across folds]")
show_summary_cols = ["seed", "setting", "backbone", "n_folds", "test_nrmse_norm_mean", "test_nrmse_norm_std", "test_mae_norm_mean", "test_mae_norm_std", "test_r2_norm_mean", "test_r2_norm_std"]
show_summary_cols = [c for c in show_summary_cols if c in summary_df.columns]
print(summary_df[show_summary_cols].to_string(index=False))
print(f"\n[SAVED] {fold_csv}")
print(f"[SAVED] {summary_csv}")
PY
}

for SEED_VALUE in ${SEEDS_STR}; do
  run_one_seed "${SEED_VALUE}"
done

# Aggregate all seeds.
python - <<'PY'
from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE = Path("runs_pvod_loso_val2_seed5")
fold_paths = sorted(BASE.glob("seed*/loso_fold_results.csv"))
summary_paths = sorted(BASE.glob("seed*/loso_summary_mean_std.csv"))

if not fold_paths:
    raise RuntimeError(f"No per-seed fold results found under {BASE}")

fold_df = pd.concat([pd.read_csv(p) for p in fold_paths], ignore_index=True)
fold_out = BASE / "loso_all_seeds_fold_results.csv"
fold_df.to_csv(fold_out, index=False)

metric_cols = [
    "test_rmse_norm", "test_mae_norm", "test_nrmse_norm", "test_r2_norm",
    "test_station_rmse_norm", "test_station_mae_norm", "test_station_nrmse_norm", "test_station_r2_norm",
    "daytime_nrmse_norm", "cloudy_nrmse_norm", "peak_nrmse_norm", "ramp_nrmse_norm",
]
metric_cols = [c for c in metric_cols if c in fold_df.columns]

rows = []
for (setting, backbone), g in fold_df.groupby(["setting", "backbone"]):
    row = {
        "setting": setting,
        "backbone": backbone,
        "n_runs": len(g),
        "n_seeds": g["seed"].nunique(),
        "n_test_stations": g["test_station"].nunique(),
    }
    for c in metric_cols:
        vals = pd.to_numeric(g[c], errors="coerce")
        row[f"{c}_mean"] = vals.mean()
        row[f"{c}_std"] = vals.std(ddof=1)
    rows.append(row)

summary_df = pd.DataFrame(rows)
if not summary_df.empty and "test_nrmse_norm_mean" in summary_df.columns:
    summary_df = summary_df.sort_values("test_nrmse_norm_mean")

summary_out = BASE / "loso_all_seeds_summary_mean_std.csv"
summary_df.to_csv(summary_out, index=False)

print("\n[ALL-SEED SUMMARY: mean ± std across seeds × LOSO folds]")
show_cols = ["setting", "backbone", "n_runs", "n_seeds", "n_test_stations", "test_nrmse_norm_mean", "test_nrmse_norm_std", "test_mae_norm_mean", "test_mae_norm_std", "test_r2_norm_mean", "test_r2_norm_std"]
show_cols = [c for c in show_cols if c in summary_df.columns]
print(summary_df[show_cols].to_string(index=False))
print(f"\n[SAVED] {fold_out}")
print(f"[SAVED] {summary_out}")
PY

echo "[DONE] Sequential 5-seed LOSO evaluation completed."
