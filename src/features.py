"""Pre-game matchup features.

Design rules
------------
* Every feature for a game in (season, week) uses only information from games
  *strictly before* that week. Week 1 therefore uses last season's state.
* All features are "edges": positive means better for the HOME team. That keeps
  the model's coefficients interpretable (they should all be >= 0) and makes the
  explanation text trivial to get right.
* Everything is precomputed once per FeatureStore, so building a whole
  training set is fast (no per-game recomputation).
"""
from __future__ import annotations

import math
from bisect import bisect_right
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .teams import canon

FEATURE_COLS = [
    "off_epa_ewm_edge",     # home off EPA/play (recency weighted) - away
    "def_epa_ewm_edge",     # away def EPA allowed - home def EPA allowed (lower allowed = better)
    "net_epa_season_edge",  # season-to-date net EPA/play, shrunk toward last season
    "to_margin_edge",       # turnover margin per game, shrunk toward 0
    "elo_edge",             # Elo rating difference (home - away), no home-field adjustment
    "injury_edge",          # away injury burden - home injury burden (OUT players only)
    "neutral_site",         # 1 if the game is at a neutral site
]

EWM_SPAN = 8          # games; spans seasons so week 1 reflects last year's form
SHRINK_GAMES = 4      # pseudo-games of prior used for season-to-date stats

# Positional value weights (rough "moves the needle" weights)
POS_WEIGHTS: Dict[str, float] = {
    "QB": 3.5,
    "OT": 1.2, "T": 1.2, "G": 0.9, "C": 0.8, "OL": 1.0,
    "WR": 1.1, "TE": 0.8, "RB": 0.7,
    "EDGE": 1.2, "DE": 1.1, "OLB": 1.0, "DT": 0.9,
    "CB": 1.2, "S": 0.9, "FS": 0.9, "SS": 0.9,
    "LB": 0.8, "ILB": 0.8,
    "K": 0.2, "P": 0.2,
}
POS_ALIASES = {"HB": "RB", "FB": "RB", "NT": "DT", "DB": "S"}
OUT_TOKENS = {"out"}  # only OUT counts as not playing
DEFAULT_SNAP_PCT = 35.0  # caliber assumed for a player we can't match to snap counts


# =============================================================================
# Team form (EPA, turnovers)
# =============================================================================
def build_weekly_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season, week) with offensive/defensive EPA and turnovers."""
    df = pbp[[c for c in ("season", "week", "posteam", "defteam", "epa", "play_type",
                          "interception", "fumble_lost") if c in pbp.columns]]
    df = df[df["play_type"].isin(["run", "pass"])]
    df = df.dropna(subset=["posteam", "defteam", "epa"])

    zero = pd.Series(0.0, index=df.index)
    itc = df["interception"].fillna(0).astype(float) if "interception" in df.columns else zero
    fum = df["fumble_lost"].fillna(0).astype(float) if "fumble_lost" in df.columns else zero
    df = df.assign(turnover=itc + fum)

    off = df.groupby(["season", "week", "posteam"]).agg(
        off_epa=("epa", "mean"), to_committed=("turnover", "sum"))
    off.index.names = ["season", "week", "team"]
    dfn = df.groupby(["season", "week", "defteam"]).agg(
        def_epa=("epa", "mean"), to_forced=("turnover", "sum"))
    dfn.index.names = ["season", "week", "team"]

    w = off.join(dfn, how="outer").fillna(0.0).reset_index()
    w = w.sort_values(["team", "season", "week"]).reset_index(drop=True)
    w["net"] = w["off_epa"] - w["def_epa"]
    w["to_margin"] = w["to_forced"] - w["to_committed"]
    return w


class TeamForm:
    """Pre-game team state, looked up by (team, season, week)."""

    _ZERO = {"off_ewm": 0.0, "def_ewm": 0.0, "net_season": 0.0, "to_margin": 0.0}

    def __init__(self, pbp: pd.DataFrame, span: int = EWM_SPAN, shrink: float = SHRINK_GAMES):
        self.shrink = float(shrink)
        w = build_weekly_table(pbp)

        # recency-weighted form across seasons (so week 1 carries last year's form)
        w["off_ewm"] = w.groupby("team")["off_epa"].transform(
            lambda s: s.ewm(span=span, adjust=False).mean())
        w["def_ewm"] = w.groupby("team")["def_epa"].transform(
            lambda s: s.ewm(span=span, adjust=False).mean())

        # season-to-date running totals
        grp = w.groupby(["team", "season"])
        w["cum_net"] = grp["net"].cumsum()
        w["cum_to"] = grp["to_margin"].cumsum()
        w["n"] = grp.cumcount() + 1

        # previous season's mean net EPA (the prior we shrink toward)
        season_mean = grp["net"].mean().to_dict()
        w["prior_net"] = [season_mean.get((t, s - 1), 0.0) for t, s in zip(w["team"], w["season"])]
        w["t"] = w["season"].astype(int) * 100 + w["week"].astype(int)

        self._hist: Dict[str, Dict[str, np.ndarray]] = {}
        for team, g in w.groupby("team"):
            self._hist[team] = {
                c: g[c].to_numpy() for c in
                ("t", "season", "off_ewm", "def_ewm", "cum_net", "cum_to", "n", "prior_net")
            }
        self.teams = set(self._hist)

    def state(self, team: str, season: int, week: int) -> Dict[str, float]:
        """State after the team's last game strictly before (season, week)."""
        h = self._hist.get(team)
        if h is None:
            return dict(self._ZERO)
        i = int(np.searchsorted(h["t"], season * 100 + week, side="left")) - 1
        if i < 0:
            return dict(self._ZERO)
        n = float(h["n"][i])
        if int(h["season"][i]) == season:
            k = self.shrink
            net = (h["cum_net"][i] + k * h["prior_net"][i]) / (n + k)
            to = h["cum_to"][i] / (n + k)
        else:  # no game yet this season: season-to-date == last season's full mean
            net = h["cum_net"][i] / n
            to = 0.0
        return {
            "off_ewm": float(h["off_ewm"][i]),
            "def_ewm": float(h["def_ewm"][i]),
            "net_season": float(net),
            "to_margin": float(to),
        }


# =============================================================================
# Elo (carries across seasons)
# =============================================================================
class EloRatings:
    """Elo built from schedule results across seasons.

    * margin-of-victory multiplier, home-field bump (none for neutral sites)
    * ratings regress 1/3 toward the mean between seasons
    * ``at(season, week)`` returns ratings as of the START of that week.
    """

    def __init__(self, schedules: pd.DataFrame, k: float = 20.0, hfa: float = 48.0,
                 revert: float = 1 / 3, base: float = 1500.0):
        s = schedules.sort_values(["season", "week"], kind="mergesort")
        ratings: Dict[str, float] = {}
        snaps: Dict[tuple, Dict[str, float]] = {}
        prev_season = None

        for (season, week), games in s.groupby(["season", "week"], sort=True):
            season, week = int(season), int(week)
            if prev_season is not None and season != prev_season:
                for t in ratings:
                    ratings[t] = base + (1.0 - revert) * (ratings[t] - base)
            prev_season = season
            snaps[(season, week)] = dict(ratings)  # state BEFORE this week's games

            for g in games.itertuples(index=False):
                hs, as_ = g.home_score, g.away_score
                if pd.isna(hs) or pd.isna(as_):
                    continue
                home, away = g.home_team, g.away_team
                rh, ra = ratings.get(home, base), ratings.get(away, base)
                neutral = str(getattr(g, "location", "Home")).lower() == "neutral"
                diff = (rh + (0.0 if neutral else hfa)) - ra
                exp_h = 1.0 / (1.0 + 10 ** (-diff / 400.0))
                if hs > as_:
                    score_h, winner_diff = 1.0, diff
                elif hs < as_:
                    score_h, winner_diff = 0.0, -diff
                else:
                    score_h, winner_diff = 0.5, 0.0
                mov = abs(hs - as_)
                mult = math.log(mov + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2) if mov > 0 else 1.0
                delta = k * mult * (score_h - exp_h)
                ratings[home] = rh + delta
                ratings[away] = ra - delta

        self._snaps = snaps
        self._keys = sorted(snaps)
        self._final = dict(ratings)
        self._base = base

    def at(self, season: int, week: int) -> Dict[str, float]:
        key = (int(season), int(week))
        if key in self._snaps:
            return self._snaps[key]
        if not self._keys or key > self._keys[-1]:
            return self._final
        i = bisect_right(self._keys, key) - 1
        return self._snaps[self._keys[i]] if i >= 0 else {}

    def rating(self, team: str, season: int, week: int) -> float:
        return self.at(season, week).get(team, self._base)


# =============================================================================
# Injuries (OUT only)
# =============================================================================
def _normalize_pos(pos) -> str:
    if not isinstance(pos, str):
        return "UNK"
    pos = pos.strip().upper()
    return POS_ALIASES.get(pos, pos)


class InjuryModel:
    """Positional + snap-share weighted burden of OUT players, per (season, week, team).

    Snap share ("caliber") uses the 6 weeks BEFORE the game week only, so a
    player who was ruled out is not penalised by the zero snaps he plays that week.
    """

    def __init__(self, injuries: Optional[pd.DataFrame] = None,
                 snaps: Optional[pd.DataFrame] = None,
                 live_key: Optional[tuple] = None):
        inj = injuries if injuries is not None else pd.DataFrame(
            columns=["season", "week", "team", "player_name", "position", "status"])
        inj = inj.copy()
        inj["status"] = inj["status"].astype(str).str.lower().str.strip()
        self._weeks = set(zip(inj["season"].astype(int), inj["week"].astype(int)))
        out = inj[inj["status"].isin(OUT_TOKENS)]
        self._out = {(int(s), int(w)): g for (s, w), g in out.groupby(["season", "week"])}
        self._snaps = ({int(s): g for s, g in snaps.groupby("season")}
                       if snaps is not None and len(snaps) else {})
        self.live_key = live_key
        self._score_cache: Dict[tuple, Dict[str, float]] = {}
        self._snap_cache: Dict[tuple, Dict[tuple, float]] = {}
        self._player_cache: Dict[tuple, Dict[str, list]] = {}

    def has_data(self, season: int, week: int) -> bool:
        return (int(season), int(week)) in self._weeks

    def note(self, season: int, week: int) -> str:
        key = (int(season), int(week))
        if not self.has_data(*key):
            return "No injury report available for this week, so injuries are treated as even."
        if self.live_key == key:
            return "Injuries: ESPN's current report (OUT players only)."
        return "Injuries: official weekly report (OUT players only)."

    def _caliber(self, season: int, week: int) -> Dict[tuple, float]:
        key = (season, week)
        if key in self._snap_cache:
            return self._snap_cache[key]
        sn = self._snaps.get(season)
        avg: Dict[tuple, float] = {}
        if sn is not None:
            m = sn[(sn["week"] < week) & (sn["week"] >= week - 6) & (sn["snap_pct"] > 0)]
            avg = m.groupby(["team", "player_name"])["snap_pct"].mean().to_dict()
        self._snap_cache[key] = avg
        return avg

    def _table(self, season: int, week: int) -> Dict[str, float]:
        key = (season, week)
        if key in self._score_cache:
            return self._score_cache[key]
        out = self._out.get(key)
        scores: Dict[str, float] = {}
        players: Dict[str, list] = {}
        if out is not None and len(out):
            caliber = self._caliber(season, week)
            for r in out.itertuples(index=False):
                pos = _normalize_pos(r.position)
                w = POS_WEIGHTS.get(pos, 0.6)
                name = str(r.player_name).strip()
                pct = caliber.get((r.team, name), DEFAULT_SNAP_PCT)
                mult = float(np.clip(pct / 50.0, 0.5, 2.0))
                impact = w * mult
                scores[r.team] = scores.get(r.team, 0.0) + impact
                players.setdefault(r.team, []).append(
                    {"name": name, "position": pos, "impact": float(impact)})
            for v in players.values():
                v.sort(key=lambda d: -d["impact"])
        self._score_cache[key] = scores
        self._player_cache[key] = players
        return scores

    def score(self, season: int, week: int, team: str) -> float:
        return self._table(int(season), int(week)).get(team, 0.0)

    def out_players(self, season: int, week: int, team: str,
                    min_impact: float = 1.2, limit: int = 5) -> list:
        """Notable OUT players for a team (biggest estimated impact first)."""
        key = (int(season), int(week))
        self._table(*key)
        return [p for p in self._player_cache[key].get(team, [])
                if p["impact"] >= min_impact][:limit]


# =============================================================================
# Facade
# =============================================================================
class FeatureStore:
    def __init__(self, pbp: pd.DataFrame, schedules: pd.DataFrame,
                 injuries: Optional[pd.DataFrame] = None,
                 snaps: Optional[pd.DataFrame] = None,
                 live_injury_key: Optional[tuple] = None):
        self.form = TeamForm(pbp)
        self.elo = EloRatings(schedules)
        self.injuries = InjuryModel(injuries, snaps, live_key=live_injury_key)
        self.teams = set(schedules["home_team"]) | set(schedules["away_team"])

    def matchup(self, season: int, week: int, home: str, away: str,
                neutral: bool = False) -> Dict[str, float]:
        season, week = int(season), int(week)
        home, away = canon(home), canon(away)
        h = self.form.state(home, season, week)
        a = self.form.state(away, season, week)
        return {
            "off_epa_ewm_edge": h["off_ewm"] - a["off_ewm"],
            "def_epa_ewm_edge": a["def_ewm"] - h["def_ewm"],
            "net_epa_season_edge": h["net_season"] - a["net_season"],
            "to_margin_edge": h["to_margin"] - a["to_margin"],
            "elo_edge": self.elo.rating(home, season, week) - self.elo.rating(away, season, week),
            "injury_edge": self.injuries.score(season, week, away) - self.injuries.score(season, week, home),
            "neutral_site": float(bool(neutral)),
        }

    def matchup_frame(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame([self.matchup(*args, **kwargs)])[FEATURE_COLS]
