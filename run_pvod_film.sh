#!/bin/bash
set -e

mkdir -p logs

python run_all.py \
  --csv datasets/PVOD_processed.csv \
  --time_col datetime \
  --target_col target_power_norm \
  --pclear_col P_clear_norm \
  --split_type station \
  --station_col station_id \
  --capacity_col capacity \
  --train_stations 0 1 2 3 4 \
  --val_stations 5 6 \
  --test_stations 7 8 9 \
  --settings direct physics_feature decomposition clear_sky_film \
  --backbones mlp tcn transformer \
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
  --eval_physics_cols \
    P_clear_norm \
    longitude \
    latitude \
    array_tilt \
    array_azimuth \
    hour_sin \
    hour_cos \
    doy_sin \
    doy_cos \
  --seq_len 24 \
  --batch_size 64 \
  --epochs 200 \
  --patience 20 \
  --lr 5e-4 \
  --hidden_dim 128 \
  --depth 4 \
  --dropout 0.1 \
  --residual_scale 0.2 \
  --residual_penalty 0.01 \
  --film_modulation_scale 0.1 \
  --device cuda \
  --out_root runs_pvod_film \
  --seed 42
