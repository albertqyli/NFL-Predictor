from __future__ import annotations

from typing import Iterable

import pandas as pd

from .features import FeatureStore


def build_dataset(store: FeatureStore, schedules: pd.DataFrame, seasons: Iterable[int],
                  game_types=("REG",)) -> pd.DataFrame:
    """One row per completed game: pre-game features + ``home_win`` label.

    Ties are dropped (they are neither a home win nor an away win).
    Week 1 is included; it uses last season's state.
    """
    seasons = [int(s) for s in seasons]
    s = schedules[schedules["season"].isin(seasons) & schedules["game_type"].isin(game_types)]
    s = s.dropna(subset=["home_score", "away_score"])
    s = s[s["home_score"] != s["away_score"]]

    rows = []
    for g in s.itertuples(index=False):
        neutral = str(getattr(g, "location", "Home")).lower() == "neutral"
        row = store.matchup(g.season, g.week, g.home_team, g.away_team, neutral)
        row.update(
            season=int(g.season), week=int(g.week),
            home_team=g.home_team, away_team=g.away_team,
            home_win=int(g.home_score > g.away_score),
        )
        rows.append(row)
    return pd.DataFrame(rows)
