"""Walk-forward evaluation.

For each test season S the model is trained ONLY on seasons < S and scored on S,
and compared with two baselines:
  * home_baseline : always predicts the training-set home win rate
  * elo_only      : logistic regression on elo_edge alone

Run on its own:   python -m src.backtest            (uses models/training_rows.csv)
                  python -m src.backtest --rebuild  (re-download and rebuild rows)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ROWS_PATH = MODELS_DIR / "training_rows.csv"
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


def make_model(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y)),
    }


def walk_forward(df: pd.DataFrame, test_seasons: Sequence[int], C: float = 0.7,
                 feature_cols: Sequence[str] = FEATURE_COLS) -> pd.DataFrame:
    """Out-of-sample predictions for every test season (columns: p_full, p_elo, p_home)."""
    parts: List[pd.DataFrame] = []
    for s in test_seasons:
        tr, te = df[df["season"] < s], df[df["season"] == s]
        if tr.empty or te.empty:
            continue
        full = make_model(C).fit(tr[list(feature_cols)].astype(float), tr["home_win"])
        elo = make_model(C).fit(tr[["elo_edge"]].astype(float), tr["home_win"])
        out = te[["season", "week", "home_team", "away_team", "home_win"]].copy()
        out["p_full"] = full.predict_proba(te[list(feature_cols)].astype(float))[:, 1]
        out["p_elo"] = elo.predict_proba(te[["elo_edge"]].astype(float))[:, 1]
        out["p_home"] = float(tr["home_win"].mean())
        parts.append(out)
    if not parts:
        raise ValueError("No test seasons had both training and test data.")
    return pd.concat(parts, ignore_index=True)


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics per model, per season and pooled ('ALL')."""
    rows = []
    groups = [(str(s), g) for s, g in preds.groupby("season")] + [("ALL", preds)]
    for label, g in groups:
        y = g["home_win"].to_numpy()
        for model, col in (("home_baseline", "p_home"), ("elo_only", "p_elo"), ("full_model", "p_full")):
            rows.append({"season": label, "model": model, **_metrics(y, g[col].to_numpy())})
    return pd.DataFrame(rows)


def calibration(preds: pd.DataFrame, col: str = "p_full",
                edges=(0.0, 0.4, 0.5, 0.6, 0.7, 1.0)) -> pd.DataFrame:
    b = pd.cut(preds[col], bins=list(edges), include_lowest=True)
    g = preds.groupby(b, observed=True).agg(
        games=("home_win", "size"), avg_pred=(col, "mean"), actual=("home_win", "mean"))
    return g.reset_index().rename(columns={col: "bin"})


def tune_C(df: pd.DataFrame, test_seasons: Sequence[int], grid=C_GRID) -> float:
    """Pick C by pooled walk-forward log loss (it is cheap: the data is tiny)."""
    best_c, best = None, float("inf")
    for c in grid:
        p = walk_forward(df, test_seasons, C=c)
        ll = _metrics(p["home_win"].to_numpy(), p["p_full"].to_numpy())["log_loss"]
        if ll < best:
            best_c, best = c, ll
    return float(best_c)


def format_report(summary: pd.DataFrame, calib: pd.DataFrame) -> str:
    s = summary.copy()
    for c in ("accuracy", "log_loss", "brier"):
        s[c] = s[c].map(lambda v: f"{v:.3f}")
    return ("Walk-forward results (trained only on earlier seasons):\n"
            + s.to_string(index=False)
            + "\n\nCalibration of full_model (pred vs actual home-win rate):\n"
            + calib.to_string(index=False))


def default_test_seasons(df: pd.DataFrame, min_train_seasons: int = 2) -> List[int]:
    seasons = sorted(df["season"].unique())
    return [int(s) for s in seasons[min_train_seasons:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-download data and rebuild rows")
    ap.add_argument("--C", type=float, default=None, help="regularisation (default: tune)")
    args = ap.parse_args()

    if args.rebuild or not ROWS_PATH.exists():
        from .train_model import build_rows  # heavy: downloads data
        df, _ = build_rows(None)
    else:
        df = pd.read_csv(ROWS_PATH)

    test = default_test_seasons(df)
    C = args.C if args.C is not None else tune_C(df, test)
    preds = walk_forward(df, test, C=C)
    print(f"C = {C}")
    print(format_report(summarize(preds), calibration(preds)))


if __name__ == "__main__":
    main()
