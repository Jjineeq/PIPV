from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _station_num(station_id: str) -> int:
    m = re.search(r"(\d+)$", str(station_id))
    if m is None:
        raise ValueError(f"Cannot parse station number from Station_ID={station_id!r}")
    return int(m.group(1))


def _parse_numeric(value, default=np.nan) -> float:
    """Parse numeric value from strings such as 'South 33°'."""
    if pd.isna(value):
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    m = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(m.group(0)) if m else default


def _parse_azimuth_from_tilt_text(value, default: float = 180.0) -> float:
    """Parse orientation text. PVOD has values like 'South 33°'."""
    if pd.isna(value):
        return default

    text = str(value).lower()
    if "south" in text:
        return 180.0
    if "north" in text:
        return 0.0
    if "east" in text:
        return 90.0
    if "west" in text:
        return 270.0
    return default


def _read_metadata(metadata_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    required = ["Station_ID", "Capacity", "Longitude", "Latitude"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")

    meta = meta.copy()
    meta["station_id"] = meta["Station_ID"].map(_station_num).astype(int)
    meta["station_name"] = meta["Station_ID"].astype(str)
    return meta


def _compute_pclear_pvlib(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    capacity: float,
    tilt: Optional[float],
    azimuth: float,
    tz: str,
) -> pd.Series:
    """
    Clear-sky proxy from pvlib.
    Output scale is capacity-scaled and intended as a physical envelope,
    not a calibrated PVWatts plant model.
    """
    import pvlib

    times = pd.to_datetime(df["datetime"])
    if times.dt.tz is None:
        times_local = times.dt.tz_localize(tz)
    else:
        times_local = times.dt.tz_convert(tz)

    loc = pvlib.location.Location(latitude, longitude, tz=tz)
    solpos = loc.get_solarposition(times_local)
    cs = loc.get_clearsky(times_local, model="ineichen")

    surface_tilt = float(tilt) if pd.notna(tilt) else float(latitude)
    surface_azimuth = float(azimuth) if pd.notna(azimuth) else 180.0

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
    )["poa_global"]

    denom = float(np.nanpercentile(poa.to_numpy(dtype=float), 99.5))
    denom = max(denom, 1e-8)
    p_clear = capacity * np.clip(poa.to_numpy(dtype=float) / denom, 0.0, 1.2)
    return pd.Series(p_clear, index=df.index, name="P_clear")


def _compute_pclear_fallback(df: pd.DataFrame, capacity: float) -> pd.Series:
    """
    Fallback when pvlib is unavailable.
    Uses observed irradiance envelope proxy.
    """
    if "lmd_totalirrad" in df.columns:
        irr = pd.to_numeric(df["lmd_totalirrad"], errors="coerce").to_numpy(dtype=float)
    elif "nwp_globalirrad" in df.columns:
        irr = pd.to_numeric(df["nwp_globalirrad"], errors="coerce").to_numpy(dtype=float)
    else:
        irr = np.zeros(len(df), dtype=float)

    denom = float(np.nanpercentile(irr, 99.5))
    denom = max(denom, 1e-8)
    p_clear = capacity * np.clip(irr / denom, 0.0, 1.2)
    return pd.Series(p_clear, index=df.index, name="P_clear")


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    hour = dt.dt.hour.to_numpy()
    doy = dt.dt.dayofyear.to_numpy()

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def _standardize_station_frame(
    path: Path,
    station_num: int,
    meta_row: pd.Series,
    tz: str,
    use_pvlib: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["date_time", "power"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("datetime").drop_duplicates("datetime")

    capacity = _parse_numeric(meta_row["Capacity"])
    longitude = _parse_numeric(meta_row["Longitude"])
    latitude = _parse_numeric(meta_row["Latitude"])

    tilt_raw = meta_row["Array_Tilt"] if "Array_Tilt" in meta_row.index else np.nan
    array_tilt = _parse_numeric(tilt_raw, default=np.nan)
    array_azimuth = _parse_azimuth_from_tilt_text(tilt_raw, default=180.0)

    df["station_id"] = int(station_num)
    df["station_name"] = str(meta_row["Station_ID"])
    df["capacity"] = capacity
    df["longitude"] = longitude
    df["latitude"] = latitude
    df["array_tilt"] = array_tilt
    df["array_azimuth"] = array_azimuth

    rename_map = {
        "lmd_totalirrad": "meas_globalirrad",
        "lmd_temperature": "meas_temperature",
        "lmd_pressure": "meas_pressure",
        "lmd_winddirection": "meas_winddirection",
        "lmd_windspeed": "meas_windspeed",
        "nwp_globalirrad": "nwp_globalirrad",
        "nwp_temperature": "nwp_temperature",
        "nwp_pressure": "nwp_pressure",
        "nwp_winddirection": "nwp_winddirection",
        "nwp_windspeed": "nwp_windspeed",
    }

    for src, dst in rename_map.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    if "lmd_diffuseirrad" in df.columns:
        df["meas_diffuseirrad"] = pd.to_numeric(df["lmd_diffuseirrad"], errors="coerce")
        if "lmd_totalirrad" in df.columns:
            df["meas_beamirrad_proxy"] = (
                pd.to_numeric(df["lmd_totalirrad"], errors="coerce")
                - pd.to_numeric(df["lmd_diffuseirrad"], errors="coerce")
            ).clip(lower=0.0)

    df["target_power"] = pd.to_numeric(df["power"], errors="coerce")
    df["target_power_norm"] = df["target_power"] / max(capacity, 1e-8)

    try:
        if use_pvlib:
            df["P_clear"] = _compute_pclear_pvlib(
                df=df,
                latitude=latitude,
                longitude=longitude,
                capacity=capacity,
                tilt=array_tilt,
                azimuth=array_azimuth,
                tz=tz,
            )
        else:
            df["P_clear"] = _compute_pclear_fallback(df, capacity=capacity)
    except Exception as e:
        print(f"[WARN] pvlib P_clear failed for {path.name}: {e}")
        print("[WARN] using irradiance-envelope fallback P_clear.")
        df["P_clear"] = _compute_pclear_fallback(df, capacity=capacity)

    df["P_clear_norm"] = df["P_clear"] / max(capacity, 1e-8)
    df = _add_time_features(df)

    front_cols = [
        "datetime", "station_id", "station_name",
        "target_power", "target_power_norm", "power",
        "P_clear", "P_clear_norm",
        "capacity", "longitude", "latitude", "array_tilt", "array_azimuth",
        "meas_globalirrad", "meas_temperature", "meas_pressure", "meas_winddirection", "meas_windspeed",
        "nwp_globalirrad", "nwp_temperature", "nwp_pressure", "nwp_winddirection", "nwp_windspeed",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ]
    existing_front = [c for c in front_cols if c in df.columns]
    rest = [c for c in df.columns if c not in existing_front and c != "date_time"]
    return df[existing_front + rest]


def build_pvod_processed_csv(
    datasets_dir: str | Path = "datasets",
    out_csv: str | Path | None = None,
    metadata_name: str = "metadata.csv",
    station_glob: str = "station*.csv",
    tz: str = "Asia/Shanghai",
    use_pvlib: bool = True,
) -> tuple[pd.DataFrame, dict]:
    datasets_dir = Path(datasets_dir)
    if out_csv is None:
        out_csv = datasets_dir / "PVOD_processed.csv"
    else:
        out_csv = Path(out_csv)

    metadata_path = datasets_dir / metadata_name
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    meta = _read_metadata(metadata_path)
    meta_by_station = {int(r["station_id"]): r for _, r in meta.iterrows()}

    station_paths = sorted(datasets_dir.glob(station_glob))
    station_paths = [p for p in station_paths if p.name != metadata_name]
    if not station_paths:
        raise FileNotFoundError(f"No station files found: {datasets_dir / station_glob}")

    frames = []
    for path in station_paths:
        m = re.search(r"station(\d+)\.csv$", path.name)
        if m is None:
            print(f"[SKIP] cannot parse station id: {path.name}")
            continue

        station_num = int(m.group(1))
        if station_num not in meta_by_station:
            print(f"[SKIP] no metadata for station{station_num:02d}: {path.name}")
            continue

        print(f"[LOAD] {path.name}")
        frame = _standardize_station_frame(
            path=path,
            station_num=station_num,
            meta_row=meta_by_station[station_num],
            tz=tz,
            use_pvlib=use_pvlib,
        )
        frames.append(frame)

    if not frames:
        raise RuntimeError("No station data was loaded.")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = (
        out.groupby("station_id", group_keys=False)[numeric_cols]
        .apply(lambda x: x.interpolate(limit_direction="both").ffill().bfill())
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    common_measured_cols = [
        "meas_globalirrad", "meas_temperature", "meas_pressure", "meas_winddirection", "meas_windspeed",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ]
    common_nwp_cols = [
        "nwp_globalirrad", "nwp_temperature", "nwp_pressure", "nwp_winddirection", "nwp_windspeed",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ]
    physics_cols = [
        "P_clear", "P_clear_norm", "capacity", "longitude", "latitude", "array_tilt", "array_azimuth",
    ]

    summary = {
        "out_csv": str(out_csv),
        "n_rows": int(len(out)),
        "stations": sorted(out["station_id"].dropna().astype(int).unique().tolist()),
        "datetime_min": str(out["datetime"].min()),
        "datetime_max": str(out["datetime"].max()),
        "target_col": "target_power",
        "target_norm_col": "target_power_norm",
        "pclear_col": "P_clear",
        "station_col": "station_id",
        "recommended_weather_cols": common_measured_cols,
        "recommended_eval_weather_cols": common_nwp_cols,
        "recommended_physics_cols": physics_cols,
        "note": "weather_cols and eval_weather_cols are aligned by order. Use measured columns for training and NWP columns for validation/test.",
    }

    meta_out = out_csv.with_suffix(".meta.json")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {out_csv}")
    print(f"[SAVED] {meta_out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return out, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets_dir", type=str, default="datasets")
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--metadata_name", type=str, default="metadata.csv")
    p.add_argument("--station_glob", type=str, default="station*.csv")
    p.add_argument("--tz", type=str, default="Asia/Shanghai")
    p.add_argument("--no_pvlib", action="store_true", help="Use irradiance-envelope fallback instead of pvlib clear-sky.")
    args = p.parse_args()

    build_pvod_processed_csv(
        datasets_dir=args.datasets_dir,
        out_csv=args.out_csv,
        metadata_name=args.metadata_name,
        station_glob=args.station_glob,
        tz=args.tz,
        use_pvlib=not args.no_pvlib,
    )


if __name__ == "__main__":
    main()