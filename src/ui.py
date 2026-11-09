"""HTML/CSS building blocks for the Streamlit page (pure functions, easy to preview)."""
from __future__ import annotations

import html
from typing import Dict, List

# Approximate primary brand colours, chosen to stay legible on a light page.
TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#C83803", "CIN": "#FB4F14", "CLE": "#FF3C00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#A71930", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LAC": "#0080C6", "LAR": "#003594", "LV": "#7B8794", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#BFA25A", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#E0A100", "SEA": "#2E8B2E",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#5A1414",
}
NEUTRAL_GREY = "#6B7280"


def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def bar_colors(home: str, away: str):
    """Team colours for the probability bar; if the two are too alike, grey out the away side."""
    hc, ac = TEAM_COLORS.get(home, "#1D4ED8"), TEAM_COLORS.get(away, NEUTRAL_GREY)
    d = sum((x - y) ** 2 for x, y in zip(_rgb(hc), _rgb(ac))) ** 0.5
    return hc, (NEUTRAL_GREY if d < 90 else ac)


CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {background: #F4F6FA;}
.block-container {max-width: 860px; padding-top: 1.6rem; padding-bottom: 3rem;}
.hero {background: linear-gradient(135deg, #0B1F3A 0%, #12398A 60%, #1D5FD6 100%);
  border-radius: 20px; padding: 30px 32px; color: #fff; margin-bottom: 22px;
  box-shadow: 0 8px 24px rgba(11,31,58,.22);}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: #fff;}
.hero p {margin: 8px 0 0; font-size: 1.02rem; opacity: .85; color: #fff;}
.card {background: #fff; border: 1px solid #E5E9F2; border-radius: 18px; padding: 22px 24px;
  margin: 18px 0; box-shadow: 0 2px 10px rgba(15,23,42,.05); color: #0F172A;}
.card h3 {margin: 0 0 12px; font-size: 1.05rem; font-weight: 700; color: #0F172A;}
.matchup {display: flex; align-items: center; justify-content: space-between; gap: 8px;}
.side {flex: 1; text-align: center;}
.side img {width: 96px; height: 96px; object-fit: contain;}
.abbr {font-size: 1.35rem; font-weight: 800; margin-top: 6px; color: #0F172A;}
.tname {font-size: .86rem; color: #64748B;}
.role {display: inline-block; margin-top: 6px; font-size: .68rem; font-weight: 700; letter-spacing: .08em;
  padding: 2px 10px; border-radius: 999px; background: #EEF2FF; color: #3730A3;}
.pct {font-size: 2.6rem; font-weight: 800; letter-spacing: -.03em; margin-top: 10px; line-height: 1; color: #0F172A;}
.pctlabel {font-size: .74rem; color: #94A3B8; text-transform: uppercase; letter-spacing: .08em; margin-top: 4px;}
.vs {font-weight: 800; color: #CBD5E1; font-size: 1.1rem;}
.bar {display: flex; height: 16px; border-radius: 999px; overflow: hidden; margin: 22px 0 6px; background: #E2E8F0;}
.bar div {height: 100%;}
.winner {margin-top: 16px; text-align: center; background: #ECFDF3; color: #05603A; border: 1px solid #ABEFC6;
  border-radius: 12px; padding: 12px 16px; font-size: 1.02rem;}
.winner b {font-weight: 800;}
.reasons {margin: 0; padding-left: 20px; line-height: 1.65; color: #1E293B;}
.reasons li {margin: 3px 0;}
.inj-sum {margin: 0 0 14px; color: #1E293B; line-height: 1.5;}
.inj-grid {display: flex; gap: 14px; flex-wrap: wrap;}
.inj-col {flex: 1; min-width: 220px; background: #FFF7F5; border: 1px solid #FDDCD3; border-radius: 12px; padding: 12px 14px;}
.inj-col h4 {margin: 0 0 8px; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: #B42318;}
.inj-col ul {list-style: none; margin: 0; padding: 0;}
.inj-col li {padding: 4px 0; color: #1E293B; font-size: .95rem;}
.inj-col .none {color: #64748B; font-size: .9rem;}
.pos {display: inline-block; min-width: 34px; text-align: center; font-size: .7rem; font-weight: 700;
  background: #B42318; color: #fff; border-radius: 6px; padding: 2px 6px; margin-right: 8px;}
.hint {text-align: center; color: #64748B; padding: 28px 10px;}
.fine {font-size: .78rem; color: #94A3B8; text-align: center; margin-top: 26px; line-height: 1.5;}
@media (max-width: 640px) {
  .hero {padding: 22px 20px;} .hero h1 {font-size: 1.6rem;}
  .side img {width: 64px; height: 64px;} .pct {font-size: 1.9rem;} .card {padding: 18px 16px;}
}
</style>
"""


def hero_html() -> str:
    return ('<div class="hero"><h1>NFL Matchup Predictor</h1>'
            '<p>Win probabilities built from team efficiency, Elo ratings and key injuries.</p></div>')


def _flat(s: str) -> str:
    """Collapse to one line so Markdown never treats indented HTML as a code block."""
    return "".join(line.strip() for line in s.splitlines())


def matchup_html(home: str, away: str, home_p: float, away_p: float,
                 names: Dict[str, str], logos: Dict[str, str], winner: str,
                 neutral: bool = False) -> str:
    hc, ac = bar_colors(home, away)
    e = html.escape

    def side(code, p, role):
        return (f'<div class="side"><img src="{e(logos.get(code, ""))}" alt="{e(code)}">'
                f'<div class="abbr">{e(code)}</div><div class="tname">{e(names.get(code, code))}</div>'
                f'<div class="role">{role}</div>'
                f'<div class="pct">{p:.1f}%</div><div class="pctlabel">win probability</div></div>')

    return _flat(f"""
    <div class="card">
      <div class="matchup">{side(home, home_p, "NEUTRAL SITE" if neutral else "HOME")}
        <div class="vs">VS</div>{side(away, away_p, "AWAY")}</div>
      <div class="bar"><div style="width:{home_p:.2f}%;background:{hc}"></div>
        <div style="width:{away_p:.2f}%;background:{ac}"></div></div>
      <div class="winner">Projected winner: <b>{e(names.get(winner, winner))}</b></div>
    </div>""")


def reasons_html(reasons: List[str]) -> str:
    items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
    return _flat(f'<div class="card"><h3>Why the model leans this way</h3><ul class="reasons">{items}</ul></div>')


def injuries_html(home: str, away: str, inj: dict, winner: str) -> str:
    """Key OUT players, shown only when injuries move the result."""
    e = html.escape
    swing = inj["swing_pts"]
    favoured = home if swing > 0 else away
    summary = (f"Injuries move this game by about <b>{abs(swing):.1f} points</b> of win probability "
               f"toward <b>{e(favoured)}</b>.")
    if inj.get("flips_winner"):
        other = away if winner == home else home
        summary += f" Without them, <b>{e(other)}</b> would be the projected winner."

    def col(code, players):
        if players:
            lis = "".join(f'<li><span class="pos">{e(p["position"])}</span>{e(p["name"])}</li>' for p in players)
        else:
            lis = '<li class="none">No key players out</li>'
        return f'<div class="inj-col"><h4>{e(code)} &middot; Out</h4><ul>{lis}</ul></div>'

    op = inj["out_players"]
    return _flat(f"""
    <div class="card"><h3>Key injuries</h3><p class="inj-sum">{summary}</p>
      <div class="inj-grid">{col(home, op["home"])}{col(away, op["away"])}</div></div>""")


def hint_html(msg: str) -> str:
    return f'<div class="card hint">{html.escape(msg)}</div>'


def footer_html() -> str:
    return ('<div class="fine">Probabilities are model estimates for entertainment and analysis, '
            'not betting advice. Injury impact counts players ruled OUT only.</div>')
