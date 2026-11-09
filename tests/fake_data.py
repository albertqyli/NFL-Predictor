"""Synthetic league with known team strengths, for offline tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.teams import CANON_TEAMS

POSITIONS = ["QB", "WR", "WR", "OT", "CB", "DE", "LB", "S", "RB", "TE", "G", "DT"]


def make_league(seasons=(2019, 2020, 2021, 2022, 2023, 2024), seed=0, weeks=18):
    rng = np.random.default_rng(seed)
    teams = list(CANON_TEAMS)
    strength = {t: rng.normal(0, 1.0) for t in teams}
    sched_rows, pbp_rows, inj_rows, snap_rows = [], [], [], []

    for season in seasons:
        # strengths drift slowly between seasons
        for t in teams:
            strength[t] = 0.7 * strength[t] + rng.normal(0, 0.6)
        for week in range(1, weeks + 1):
            order = rng.permutation(teams)
            out_qb = set()
            for t in teams:
                for i, pos in enumerate(POSITIONS):
                    name = f"{t}_P{i}"
                    out = rng.random() < (0.04 if pos != "QB" else 0.06)
                    if out:
                        inj_rows.append(dict(season=season, week=week, team=t, player_name=name,
                                             position=pos, status="out"))
                        if pos == "QB":
                            out_qb.add(t)
                    elif rng.random() < 0.05:
                        inj_rows.append(dict(season=season, week=week, team=t, player_name=name,
                                             position=pos, status="questionable"))
                    if not out:
                        snap_rows.append(dict(season=season, week=week, team=t, player_name=name,
                                              snap_pct=float(np.clip(rng.normal(70, 25), 5, 100))))
            for i in range(0, 32, 2):
                h, a = order[i], order[i + 1]
                sh = strength[h] - (1.5 if h in out_qb else 0.0)
                sa = strength[a] - (1.5 if a in out_qb else 0.0)
                margin = 3.5 * (sh - sa) + 2.0 + rng.normal(0, 12)
                hs = int(max(0, round(23 + margin / 2)))
                as_ = int(max(0, round(23 - margin / 2)))
                sched_rows.append(dict(season=season, game_type="REG", week=week, home_team=h,
                                       away_team=a, home_score=hs, away_score=as_,
                                       location="Neutral" if (week == 5 and i == 0) else "Home"))
                for team, opp, s_o, s_d in ((h, a, sh, sa), (a, h, sa, sh)):
                    for _ in range(40):
                        play = "pass" if rng.random() < 0.58 else "run"
                        epa = 0.06 * s_o - 0.06 * s_d + rng.normal(0, 1.3)
                        pbp_rows.append(dict(season=season, week=week, posteam=team, defteam=opp,
                                             epa=epa, play_type=play,
                                             interception=float(rng.random() < 0.015),
                                             fumble_lost=float(rng.random() < 0.01)))
                    pbp_rows.append(dict(season=season, week=week, posteam=team, defteam=opp,
                                         epa=np.nan, play_type="punt", interception=0.0, fumble_lost=0.0))
    return (pd.DataFrame(pbp_rows), pd.DataFrame(sched_rows),
            pd.DataFrame(inj_rows), pd.DataFrame(snap_rows))
