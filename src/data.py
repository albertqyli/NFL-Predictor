"""All network / nfl_data_py access lives here.

Everything else in the project takes plain DataFrames, so it can be tested
offline and you can swap data sources in one place.
"""
from __future__ import annotations

import warnings
from datetime import date
from typing import Iterable, Optional

import pandas as pd

from .teams import NAME_TO_CODE, canon

# Elo needs a long warm-up so ratings are meaningful by the first training season.
ELO_START_SEASON = 2015

PBP_COLUMNS = [
    "season", "week", "posteam", "defteam", "epa", "play_type",
    "interception", "fumble_lost",
]

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"


def default_season(today: Optional[date] = None) -> int:
    """NFL season in progress (Aug-Dec -> this year, Jan-Jul -> the season that just ended)."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def _nfl():
    import nfl_data_py as nfl  # imported lazily so tests don't need it
    return nfl


def _canon_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        mapping = {v: canon(v) for v in df[col].dropna().unique()}
        df[col] = df[col].map(mapping)
    return df


# --------------------------------------------------------------------------- pbp
def _load_one_pbp(season: int) -> pd.DataFrame:
    nfl = _nfl()
    try:
        pbp = nfl.import_pbp_data(
            [season], columns=PBP_COLUMNS, include_participation=False,
            downcast=True, cache=False,
        )
    except TypeError:  # older nfl_data_py without include_participation
        pbp = nfl.import_pbp_data([season], columns=PBP_COLUMNS, downcast=True, cache=False)
    return pbp


def load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            frames.append(_load_one_pbp(int(s)))
        except Exception as e:  # e.g. a season that has no data yet
            warnings.warn(f"Could not load play-by-play for {s}: {e}")
    if not frames:
        raise RuntimeError("No play-by-play data could be loaded.")
    pbp = pd.concat(frames, ignore_index=True)
    for c in ("posteam", "defteam"):
        _canon_col(pbp, c)
    return pbp


# ----------------------------------------------------------------------- schedule
def load_schedules(seasons: Iterable[int]) -> pd.DataFrame:
    sched = _nfl().import_schedules(list(seasons)).copy()
    if "location" not in sched.columns:
        sched["location"] = "Home"
    keep = [c for c in ("season", "game_type", "week", "gameday", "home_team", "away_team",
                        "home_score", "away_score", "location") if c in sched.columns]
    sched = sched[keep].copy()
    for c in ("home_team", "away_team"):
        _canon_col(sched, c)
    return sched.sort_values(["season", "week"]).reset_index(drop=True)


# ---------------------------------------------------------------------- injuries
def _empty_injuries() -> pd.DataFrame:
    return pd.DataFrame(columns=["season", "week", "team", "player_name", "position", "status"])


def load_injuries(seasons: Iterable[int]) -> pd.DataFrame:
    """Historical weekly injury reports. Seasons that fail to load are skipped (with a warning)."""
    nfl = _nfl()
    frames = []
    for s in seasons:
        try:
            inj = nfl.import_injuries([int(s)])
        except Exception as e:
            warnings.warn(f"Injury data unavailable for {s}: {e}")
            continue
        if inj is None or inj.empty:
            continue
        cols = {c.lower(): c for c in inj.columns}
        team = cols.get("team") or cols.get("club") or cols.get("team_abbr")
        name = cols.get("full_name") or cols.get("player_name") or cols.get("name")
        pos = cols.get("position") or cols.get("pos")
        status = cols.get("report_status") or cols.get("status") or cols.get("game_status")
        week = cols.get("week")
        season = cols.get("season")
        if not (team and name and pos and status and week):
            warnings.warn(f"Injury data for {s} has unexpected columns: {list(inj.columns)}")
            continue
        out = pd.DataFrame({
            "season": inj[season] if season else int(s),
            "week": inj[week],
            "team": inj[team],
            "player_name": inj[name],
            "position": inj[pos],
            "status": inj[status].astype(str).str.lower().str.strip(),
        })
        frames.append(out)
    if not frames:
        return _empty_injuries()
    out = pd.concat(frames, ignore_index=True)
    _canon_col(out, "team")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out.dropna(subset=["week", "team"])
    out["week"] = out["week"].astype(int)
    out["season"] = out["season"].astype(int)
    return out


def load_live_injuries(timeout: int = 15) -> Optional[pd.DataFrame]:
    """Current injury report from ESPN's JSON endpoint (NOT historical).

    Only ever used for the upcoming week at prediction time, never for training.
    Returns None on any failure. NOTE: the response shape was written from
    memory of the endpoint and is untested here, so failures degrade to "no data".
    """
    try:
        import requests
        r = requests.get(ESPN_INJURIES_URL, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        rows = []
        for t in data.get("injuries", []):
            team = NAME_TO_CODE.get(str(t.get("displayName", "")).lower())
            if not team:
                continue
            for inj in t.get("injuries", []):
                ath = inj.get("athlete", {}) or {}
                pos = (ath.get("position") or {}).get("abbreviation")
                rows.append({
                    "team": canon(team),
                    "player_name": ath.get("displayName"),
                    "position": pos,
                    "status": str(inj.get("status", "")).lower().strip(),
                })
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception as e:
        warnings.warn(f"Live injury fetch failed: {e}")
        return None


# ------------------------------------------------------------------------- snaps
def load_snaps(seasons: Iterable[int]) -> Optional[pd.DataFrame]:
    nfl = _nfl()
    frames = []
    for s in seasons:
        try:
            sn = nfl.import_snap_counts([int(s)])
        except Exception as e:
            warnings.warn(f"Snap counts unavailable for {s}: {e}")
            continue
        cols = {c.lower(): c for c in sn.columns}
        name = cols.get("player") or cols.get("player_name") or cols.get("name")
        team = cols.get("team") or cols.get("team_abbr")
        week = cols.get("week")
        off = cols.get("offense_pct") or cols.get("off_pct")
        dfn = cols.get("defense_pct") or cols.get("def_pct")
        if not (name and team and week and (off or dfn)):
            continue
        parts = []
        if off:
            parts.append(pd.to_numeric(sn[off], errors="coerce").fillna(0.0))
        if dfn:
            parts.append(pd.to_numeric(sn[dfn], errors="coerce").fillna(0.0))
        pct = parts[0] if len(parts) == 1 else pd.concat(parts, axis=1).max(axis=1)
        frames.append(pd.DataFrame({
            "season": int(s), "week": sn[week], "team": sn[team],
            "player_name": sn[name].astype(str).str.strip(), "snap_pct": pct,
        }))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    _canon_col(out, "team")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out.dropna(subset=["week"])
    out["week"] = out["week"].astype(int)
    # nflverse reports snap share as a 0-1 fraction; normalise to percent.
    if out["snap_pct"].max() <= 1.5:
        out["snap_pct"] = out["snap_pct"] * 100.0
    return out
