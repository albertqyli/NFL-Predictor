import streamlit as st

from src.data import default_season, load_schedules
from src.predict import ModelOutdatedError, load_context, load_model, predict_matchup, upcoming_week
from src.teams import TEAM_NAMES
from src.ui import CSS, footer_html, hero_html, hint_html, injuries_html, matchup_html, reasons_html

LOGO_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{}.png"
TEAM_LOGOS = {code: LOGO_URL.format(code.lower()) for code in TEAM_NAMES}

st.set_page_config(page_title="NFL Matchup Predictor", page_icon="🏈", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(hero_html(), unsafe_allow_html=True)


# ---- cached heavy lifting: data is downloaded once per season, not once per click ----
@st.cache_resource(ttl=3600, show_spinner=False)
def get_context(season: int, elo_start: int):
    return load_context(season, elo_start=elo_start)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(ttl=3600)
def get_default_week(season: int) -> int:
    try:
        return upcoming_week(load_schedules([season]), season) or 1
    except Exception:
        return 1


try:
    model, meta = get_model()
except ModelOutdatedError:
    model, meta = None, None
if model is None:
    st.markdown(hint_html("The prediction model is being updated. Please check back in a few minutes."),
                unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------------- controls
with st.container(border=True):
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        year = st.number_input("Season", min_value=2016, max_value=2100, value=default_season(), step=1)
    with c2:
        week = st.number_input("Week", min_value=1, max_value=22, value=get_default_week(int(year)), step=1)
    with c3:
        st.write("")
        neutral = st.checkbox("Neutral-site game")

    def label(code: str) -> str:
        return f"{code} — {TEAM_NAMES[code]}"

    teams = sorted(TEAM_NAMES)
    hcol, acol = st.columns(2)
    with hcol:
        home = st.selectbox("Home team", teams, index=None, placeholder="Choose home team",
                            format_func=label)
        if home:
            st.image(TEAM_LOGOS[home], width=84)
    with acol:
        away = st.selectbox("Away team", [t for t in teams if t != home], index=None,
                            placeholder="Choose away team", format_func=label)
        if away:
            st.image(TEAM_LOGOS[away], width=84)

# ----------------------------------------------------------------------------- result
if not (home and away):
    st.markdown(hint_html("Pick a home team and an away team to see the prediction."), unsafe_allow_html=True)
else:
    try:
        with st.spinner("Crunching the numbers... (the first prediction can take about a minute)"):
            ctx = get_context(int(year), meta.get("elo_start_season", 2015))
            res = predict_matchup(int(year), int(week), home, away, neutral_site=neutral,
                                  ctx=ctx, model=model, meta=meta)
    except Exception:
        st.markdown(hint_html("Couldn't load the latest NFL data right now. Please try again in a minute."),
                    unsafe_allow_html=True)
        st.stop()

    st.markdown(
        matchup_html(home, away, res["home_win_prob"] * 100, res["away_win_prob"] * 100,
                     TEAM_NAMES, TEAM_LOGOS, res["projected_winner"], neutral=neutral),
        unsafe_allow_html=True)
    st.markdown(reasons_html(res["reasons"]), unsafe_allow_html=True)
    if res["injuries"]["show"]:
        st.markdown(injuries_html(home, away, res["injuries"], res["projected_winner"]),
                    unsafe_allow_html=True)

with st.expander("About this model"):
    st.write(
        "A regularised logistic regression trained on every completed NFL regular-season game since 2019. "
        "For each game it only uses information available before kickoff: recency-weighted offensive and "
        "defensive EPA per play, season-to-date efficiency, turnover margin, an Elo rating that carries over "
        "between seasons, and the weighted impact of players ruled OUT."
    )
    bt = (meta.get("backtest") or {}).get("pooled", {})
    full, base = bt.get("full_model"), bt.get("home_baseline")
    if full and base:
        st.write(
            f"**Backtest:** trained only on earlier seasons and tested on {full['n']} later games, "
            f"it picked the winner {full['accuracy']:.1%} of the time, versus {base['accuracy']:.1%} "
            "for always picking the home team."
        )

st.markdown(footer_html(), unsafe_allow_html=True)
