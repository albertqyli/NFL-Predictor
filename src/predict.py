from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .data import (ELO_START_SEASON, load_injuries, load_live_injuries, load_pbp,
                   load_schedules, load_snaps)
from .features import FEATURE_COLS, FeatureStore
from .teams import canon

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH = MODELS_DIR / "meta.json"

FEATURE_TEXT = {
    "elo_edge": "overall team strength (Elo)",
    "off_epa_ewm_edge": "recent offensive efficiency",
    "def_epa_ewm_edge": "recent defensive efficiency",
    "net_epa_season_edge": "season-long efficiency",
    "to_margin_edge": "turnover margin",
    "injury_edge": "injuries",
}


class ModelOutdatedError(RuntimeError):
    pass


def load_model():
    """Returns (model, meta), or (None, None) if nothing has been trained yet."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    if meta.get("feature_cols") != FEATURE_COLS:
        raise ModelOutdatedError(
            "The saved model was trained with a different feature set. "
            "Re-run:  python -m src.train_model")
    return model, meta


@dataclass
class Context:
    """Everything expensive for one season, load once and reuse."""
    season: int
    store: FeatureStore
    schedules: pd.DataFrame
    next_week: Optional[int]


def upcoming_week(schedules: pd.DataFrame, season: int) -> Optional[int]:
    """First week of the season that still has unplayed games (None if the season is complete)."""
    s = schedules[schedules["season"] == season]
    pending = s[s["home_score"].isna() | s["away_score"].isna()]
    return int(pending["week"].min()) if len(pending) else None


def load_context(season: int, elo_start: int = ELO_START_SEASON, use_live_injuries: bool = True) -> Context:
    pbp = load_pbp([season - 1, season])
    sched = load_schedules(range(elo_start, season + 1))
    inj = load_injuries([season])
    snaps = load_snaps([season])
    nxt = upcoming_week(sched, season)

    live_key = None
    # Live (current) injuries are only valid for the upcoming week, never for past ones.
    if use_live_injuries and nxt is not None:
        have = (not inj.empty) and bool(((inj["season"] == season) & (inj["week"] == nxt)).any())
        if not have:
            live = load_live_injuries()
            if live is not None:
                live = live.assign(season=season, week=nxt)
                inj = pd.concat([inj, live], ignore_index=True)
                live_key = (season, nxt)
    store = FeatureStore(pbp, sched, inj, snaps, live_injury_key=live_key)
    return Context(season=season, store=store, schedules=sched, next_week=nxt)


def _explain_lines(feature_cols, raw_x, contrib, home: str, away: str, neutral: bool) -> list:
    """The largest logit contributions as short sentences. Edges are positive-for-home, so the
    sign of each contribution tells us exactly who the feature favours."""
    lines = []
    for idx in np.argsort(-np.abs(contrib)):
        f = feature_cols[idx]
        if f == "neutral_site" or f not in FEATURE_TEXT or abs(contrib[idx]) < 0.02:
            continue
        # If the fitted coefficient has the "wrong" sign (possible with overlapping features),
        # the model's effect contradicts the raw matchup; narrating it would mislead, so skip.
        if np.sign(raw_x[idx]) != np.sign(contrib[idx]):
            continue
        favours, other = (home, away) if contrib[idx] > 0 else (away, home)
        if f == "injury_edge":
            lines.append(f"{other} is missing more high-impact players who are OUT, which favors {favours}.")
        else:
            lines.append(f"{favours} has the edge in {FEATURE_TEXT[f]}.")
        if len(lines) == 3:
            break
    if not lines:
        lines.append("The teams look evenly matched on the model's inputs.")
    lines.append("This is a neutral-site game, so there is no home-field boost."
                 if neutral else f"{home} also gets the usual home-field boost.")
    return lines


def _explain(*args, **kwargs) -> str:
    return " ".join(_explain_lines(*args, **kwargs))


# An injury section is only shown when injuries move the win probability by at least this much.
INJURY_MIN_SWING_PTS = 1.0


def predict_matchup(year: int, week: int, home_team: str, away_team: str,
                    neutral_site: bool = False, ctx: Optional[Context] = None,
                    model=None, meta=None):
    if model is None:
        model, meta = load_model()
    if model is None:
        return None

    home_c, away_c = canon(home_team), canon(away_team)
    if home_c == away_c:
        raise ValueError("Home and away team must be different.")

    ctx = ctx or load_context(int(year), elo_start=meta.get("elo_start_season", ELO_START_SEASON))
    for t in (home_c, away_c):
        if t not in ctx.store.teams:
            raise ValueError(f"Unknown team code: {t}")

    cols = meta["feature_cols"]
    X = ctx.store.matchup_frame(int(year), int(week), home_team, away_team, neutral=neutral_site)
    Xv = X[cols].astype(float)

    proba_home = float(model.predict_proba(Xv)[0, 1])
    clf, scaler = model.named_steps["clf"], model.named_steps["scaler"]
    # linear term of each feature on the raw scale: coef/scale * x  (sign == who it favours)
    contrib = Xv.iloc[0].to_numpy() * (clf.coef_[0] / scaler.scale_)

    home_lbl, away_lbl = home_team.strip().upper(), away_team.strip().upper()
    reasons = _explain_lines(cols, Xv.iloc[0].to_numpy(), contrib, home_lbl, away_lbl, bool(neutral_site))

    # How much do injuries matter? Re-predict with the injury feature neutralised.
    Xn = Xv.copy()
    Xn["injury_edge"] = 0.0
    proba_no_inj = float(model.predict_proba(Xn)[0, 1])
    swing_pts = (proba_home - proba_no_inj) * 100.0          # >0: injuries help the home team
    flips = (proba_home >= 0.5) != (proba_no_inj >= 0.5)
    inj = ctx.store.injuries
    show_inj = inj.has_data(int(year), int(week)) and (abs(swing_pts) >= INJURY_MIN_SWING_PTS or flips)
    out_players = {"home": [], "away": []}
    if show_inj:
        out_players = {
            "home": inj.out_players(int(year), int(week), home_c),
            "away": inj.out_players(int(year), int(week), away_c),
        }
        if not (out_players["home"] or out_players["away"]):
            show_inj = False

    return {
        "home_win_prob": proba_home,
        "away_win_prob": 1.0 - proba_home,
        "projected_winner": home_team if proba_home >= 0.5 else away_team,
        "explanation": " ".join(reasons),
        "reasons": reasons,
        "injury_note": inj.note(int(year), int(week)),
        "injury_data_available": inj.has_data(int(year), int(week)),
        "injuries": {
            "show": bool(show_inj),
            "swing_pts": float(swing_pts),           # + favours home, - favours away
            "flips_winner": bool(flips),
            "out_players": out_players,
        },
        "factors": {f: float(c) for f, c in zip(cols, contrib)},
    }
