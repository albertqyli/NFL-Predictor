import streamlit as st

from src.data import default_season, load_schedules
from src.predict import ModelOutdatedError, load_context, load_model, predict_matchup, upcoming_week
from src.teams import TEAM_NAMES

LOGO_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{}.png"
TEAM_LOGOS = {code: LOGO_URL.format(code.lower()) for code in TEAM_NAMES}


# ---- cached heavy lifting: data is downloaded once per season, not once per click ----
@st.cache_resource(ttl=3600, show_spinner="Loading NFL data (first run takes a minute)...")
def get_context(season: int, elo_start: int):
    return load_context(season, elo_start=elo_start)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(ttl=3600)
def get_default_week(season: int) -> int:
    try:
        wk = upcoming_week(load_schedules([season]), season)
        return wk or 1
    except Exception:
        return 1


st.set_page_config(page_title="NFL Win Probability", layout="centered")
st.title("NFL Matchup Predictor")
st.caption("Win probability + projected winner + explanation (EPA form, Elo, injuries OUT-only).")

try:
    model, meta = get_model()
except ModelOutdatedError as e:
    st.error(str(e))
    st.stop()
if model is None:
    st.error("Model not found. Train it first:  `python -m src.train_model`")
    st.stop()

bt = (meta.get("backtest") or {}).get("pooled", {})
if bt:
    full, base = bt.get("full_model"), bt.get("home_baseline")
    if full and base:
        st.caption(
            f"Walk-forward backtest ({len(meta['backtest']['test_seasons'])} seasons, {full['n']} games): "
            f"model {full['accuracy']:.1%} accuracy vs. {base['accuracy']:.1%} for 'always pick the home team'."
        )

season0 = default_season()
c1, c2 = st.columns(2)
with c1:
    year = st.number_input("Season", min_value=2010, max_value=2100, value=season0, step=1)
    week = st.number_input("Week", min_value=1, max_value=22, value=get_default_week(int(year)), step=1)
    neutral = st.checkbox("Neutral-site game")

teams = sorted(TEAM_LOGOS)


def team_label(t: str) -> str:
    return f"{t} — {TEAM_NAMES.get(t, t)}"


with c2:
    hc, ac = st.columns(2)
    with hc:
        home_label = st.selectbox("Home team", ["— Select —"] + [team_label(t) for t in teams], index=0)
        home = home_label.split(" — ")[0] if " — " in home_label else ""
        if home in TEAM_LOGOS:
            st.image(TEAM_LOGOS[home], width=90)
    with ac:
        away_teams = [t for t in teams if t != home]
        away_label = st.selectbox("Away team", ["— Select —"] + [team_label(t) for t in away_teams], index=0)
        away = away_label.split(" — ")[0] if " — " in away_label else ""
        if away in TEAM_LOGOS:
            st.image(TEAM_LOGOS[away], width=90)

st.divider()

if st.button("Predict", type="primary"):
    if home not in TEAM_LOGOS or away not in TEAM_LOGOS:
        st.warning("Please select both a Home and Away team.")
        st.stop()

    try:
        ctx = get_context(int(year), meta.get("elo_start_season", 2015))
        res = predict_matchup(int(year), int(week), home, away, neutral_site=neutral,
                              ctx=ctx, model=model, meta=meta)
    except Exception as e:
        st.error(f"Could not make a prediction: {e}")
        st.stop()

    home_p, away_p = res["home_win_prob"] * 100, res["away_win_prob"] * 100

    l1, mid, l2 = st.columns([1, 1, 1])
    with l1:
        st.image(TEAM_LOGOS[home], width=110)
        st.markdown(f"### {home} (Home)")
        st.metric("Win %", f"{home_p:.1f}%")
    with mid:
        st.markdown("<h2 style='text-align:center'>vs</h2>", unsafe_allow_html=True)
    with l2:
        st.image(TEAM_LOGOS[away], width=110)
        st.markdown(f"### {away} (Away)")
        st.metric("Win %", f"{away_p:.1f}%")

    st.subheader("Projected winner")
    st.success(f"Projected winner: **{res['projected_winner']}**")

    st.subheader("Why")
    st.write(res["explanation"])
    st.caption(res["injury_note"])
    st.caption("If you change features, re-run training so models/model.joblib updates.")
