# === processed_pv_data.csv 생성 셀 ===
# 기존 250711_modeling.ipynb의 데이터 로딩 구조 기준
# 저장 결과:
# - processed_pv_data.csv
# - processed_pv_data.meta.json

from pathlib import Path
import sys

# prepare_processed_pv_csv.py가 있는 경로를 추가
sys.path.append("./")

from prepare_processed_pv_csv import build_processed_csv

processed, meta = build_processed_csv(
    data_dir="../data",
    out_csv="./processed_pv_data.csv",
    latitude=35.066767,
    longitude=127.752977,
    altitude=0.0,
    start="2021-01-01 00:00:00",
    end="2022-04-30 23:00:00",
    station_name="순천",
    fit_physics=True,
    maxiter=60,      # 빠른 확인은 20~30, 최종은 60~120 권장
    popsize=12,      # 빠른 확인은 8~10, 최종은 12~20 권장
    seed=42,
)

processed.head()
