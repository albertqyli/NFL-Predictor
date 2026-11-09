"""Team codes, names, and canonicalisation.

Different data sources disagree on some abbreviations (the Rams are "LA" in
nflverse data but "LAR" on ESPN, the Commanders are "WAS" or "WSH", ...).
Everything that enters the feature pipeline goes through ``canon`` so the
joins between play-by-play, schedules, injuries and snap counts line up.
"""
from __future__ import annotations

TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

NAME_TO_CODE = {v.lower(): k for k, v in TEAM_NAMES.items()}

_ALIASES = {
    "LAR": "LA", "STL": "LA",
    "OAK": "LV", "LVR": "LV",
    "SD": "LAC", "SDG": "LAC",
    "WSH": "WAS",
    "JAC": "JAX",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "NWE": "NE", "NOR": "NO", "GNB": "GB", "KAN": "KC", "SFO": "SF", "TAM": "TB",
}


def canon(code) -> str:
    """Map any known abbreviation to the single code used internally."""
    c = str(code).strip().upper()
    return _ALIASES.get(c, c)


CANON_TEAMS = sorted({canon(c) for c in TEAM_NAMES})
