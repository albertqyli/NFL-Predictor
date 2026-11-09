"""Train the model.

    python -m src.train_model                       # seasons 2019 .. current
    python -m src.train_model --seasons 2020 2021 2022 2023 2024 2025
    python -m src.train_model --C 0.3 --skip_backtest

All completed regular-season games in the chosen seasons are used (including the
current season so far). The season range rolls forward automatically.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd

from .backtest import (ROWS_PATH, calibration, default_test_seasons, format_report,
                       make_model, summarize, tune_C, walk_forward)
from .data import (ELO_START_SEASON, default_season, load_injuries, load_pbp,
                   load_schedules, load_snaps)
from .dataset import build_dataset
from .features import FEATURE_COLS, FeatureStore

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIRST_SEASON = 2019


def build_rows(seasons: Optional[List[int]]) -> Tuple[pd.DataFrame, List[int]]:
    seasons = seasons or list(range(FIRST_SEASON, default_season() + 1))
    seasons = sorted(set(seasons))
    print(f"Loading data for seasons {seasons} (+{seasons[0] - 1} for warm-up)...")
    pbp = load_pbp([seasons[0] - 1] + seasons)
    sched = load_schedules(range(ELO_START_SEASON, seasons[-1] + 1))
    inj = load_injuries(seasons)
    snaps = load_snaps(seasons)
    print(f"  injury rows: {len(inj)}  snap rows: {0 if snaps is None else len(snaps)}")

    print("Building features...")
    store = FeatureStore(pbp, sched, inj, snaps)
    df = build_dataset(store, sched, seasons)
    if df.empty:
        raise RuntimeError("Training dataset is empty. Check seasons and data download.")
    MODELS_DIR.mkdir(exist_ok=True)
    df.to_csv(ROWS_PATH, index=False)
    print(f"  {len(df)} games -> {ROWS_PATH}")
    return df, seasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=None)
    ap.add_argument("--C", type=float, default=None, help="regularisation; default = tune by walk-forward")
    ap.add_argument("--skip_backtest", action="store_true")
    args = ap.parse_args()

    df, seasons = build_rows(args.seasons)
    MODELS_DIR.mkdir(exist_ok=True)

    test_seasons = default_test_seasons(df)
    backtest_summary = None
    C = args.C
    if not args.skip_backtest and test_seasons:
        if C is None:
            C = tune_C(df, test_seasons)
            print(f"Chose C={C} by walk-forward log loss")
        preds = walk_forward(df, test_seasons, C=C)
        summary = summarize(preds)
        print(format_report(summary, calibration(preds)))
        backtest_summary = {
            "C": C, "test_seasons": test_seasons,
            "pooled": {r.model: {"accuracy": r.accuracy, "log_loss": r.log_loss, "brier": r.brier, "n": r.n}
                       for r in summary[summary["season"] == "ALL"].itertuples()},
        }
    C = C if C is not None else 0.3

    print("Fitting final model on all seasons...")
    model = make_model(C).fit(df[FEATURE_COLS].astype(float), df["home_win"].astype(int))
    clf, scaler = model.named_steps["clf"], model.named_steps["scaler"]
    coefs = dict(zip(FEATURE_COLS, (clf.coef_[0] / scaler.scale_).tolist()))
    print("Coefficients (per raw unit, logit scale):")
    for f, c in coefs.items():
        flag = ""
        if f != "neutral_site" and c <= 0:
            flag = "   <-- unexpected sign (edge features should help the home team)"
        print(f"  {f:22s} {c:+.4f}{flag}")

    joblib.dump(model, MODELS_DIR / "model.joblib")
    meta = {
        "feature_cols": FEATURE_COLS,
        "seasons": seasons,
        "elo_start_season": ELO_START_SEASON,
        "C": C,
        "n_rows": int(len(df)),
        "home_win_rate": float(df["home_win"].mean()),
        "coefficients": coefs,
        "backtest": backtest_summary,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model + metadata to {MODELS_DIR}")


if __name__ == "__main__":
    main()
