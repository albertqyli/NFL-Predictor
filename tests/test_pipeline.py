import numpy as np
import pandas as pd
import pytest

from fake_data import make_league
from src import predict as P
from src.backtest import (calibration, default_test_seasons, make_model, summarize,
                          tune_C, walk_forward)
from src.dataset import build_dataset
from src.features import FEATURE_COLS, EloRatings, FeatureStore

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]


@pytest.fixture(scope="module")
def league():
    return make_league(SEASONS, seed=1)


@pytest.fixture(scope="module")
def store(league):
    pbp, sched, inj, snaps = league
    return FeatureStore(pbp, sched, inj, snaps)


@pytest.fixture(scope="module")
def dataset(league, store):
    _, sched, _, _ = league
    return build_dataset(store, sched, SEASONS)


# ------------------------------------------------------------------ leakage
@pytest.mark.parametrize("season,week", [(2021, 1), (2021, 7), (2022, 18)])
def test_features_ignore_the_future(league, store, season, week):
    """Rewrite everything at/after (season, week): features for that game must not move."""
    pbp, sched, inj, snaps = (d.copy() for d in league)
    row = sched[(sched.season == season) & (sched.week == week)].iloc[0]
    before = store.matchup(season, week, row.home_team, row.away_team)

    after_mask = lambda d: (d.season > season) | ((d.season == season) & (d.week >= week))
    # scramble results, EPA, and later injury/snap data
    rng = np.random.default_rng(99)
    m = after_mask(sched)
    sched.loc[m, "home_score"] = rng.integers(0, 50, m.sum())
    sched.loc[m, "away_score"] = rng.integers(0, 50, m.sum())
    m = after_mask(pbp)
    pbp.loc[m, "epa"] = pbp.loc[m, "epa"] * -5 + 3
    pbp.loc[m, "interception"] = 1.0
    # injuries for the game week itself are legitimately known pre-game; later weeks are not
    later_inj = (inj.season > season) | ((inj.season == season) & (inj.week > week))
    inj = inj[~later_inj]
    snaps.loc[after_mask(snaps), "snap_pct"] = 1.0

    after = FeatureStore(pbp, sched, inj, snaps).matchup(season, week, row.home_team, row.away_team)
    for k in FEATURE_COLS:
        assert after[k] == pytest.approx(before[k]), k


def test_week1_uses_previous_season_not_zeros(store, league):
    _, sched, _, _ = league
    g = sched[(sched.season == 2021) & (sched.week == 1)].iloc[0]
    f = store.matchup(2021, 1, g.home_team, g.away_team)
    assert abs(f["off_epa_ewm_edge"]) > 1e-9 and abs(f["elo_edge"]) > 1e-6


def test_week1_changes_when_last_season_changes(league):
    pbp, sched, inj, snaps = league
    g = sched[(sched.season == 2021) & (sched.week == 1)].iloc[0]
    a = FeatureStore(pbp, sched, inj, snaps).matchup(2021, 1, g.home_team, g.away_team)
    p2 = pbp.copy()
    m = (p2.season == 2020) & (p2.posteam == g.home_team)
    p2.loc[m, "epa"] += 1.0
    b = FeatureStore(p2, sched, inj, snaps).matchup(2021, 1, g.home_team, g.away_team)
    assert b["off_epa_ewm_edge"] > a["off_epa_ewm_edge"] + 0.1


# --------------------------------------------------------------------- elo
def test_elo_carries_over_and_regresses(league):
    _, sched, _, _ = league
    elo = EloRatings(sched)
    late = elo.at(2020, 18)
    wk1 = elo.at(2021, 1)
    spread_late = np.std(list(late.values()))
    spread_wk1 = np.std(list(wk1.values()))
    assert spread_wk1 > 5                 # not reset to 1500
    assert spread_wk1 < spread_late * 1.2  # regressed toward the mean (roughly)


def test_elo_at_week_excludes_that_weeks_games(league):
    _, sched, _, _ = league
    g = sched[(sched.season == 2022) & (sched.week == 3)].iloc[0]
    s2 = sched.copy()
    i = s2[(s2.season == 2022) & (s2.week == 3) & (s2.home_team == g.home_team)].index[0]
    s2.loc[i, ["home_score", "away_score"]] = [0, 60]
    assert EloRatings(sched).rating(g.home_team, 2022, 3) == EloRatings(s2).rating(g.home_team, 2022, 3)


# ----------------------------------------------------------- dataset / model
def test_dataset_shape_and_week1_included(dataset):
    assert set(FEATURE_COLS) <= set(dataset.columns)
    assert (dataset.week == 1).any()
    assert dataset[FEATURE_COLS].notna().all().all()
    assert dataset.neutral_site.sum() > 0
    assert 0.4 < dataset.home_win.mean() < 0.75


def test_injury_feature_not_all_zero_and_signed(dataset):
    assert (dataset.injury_edge != 0).mean() > 0.3


def test_model_learns_sane_signs_and_beats_baseline(dataset):
    m = make_model(0.3).fit(dataset[FEATURE_COLS], dataset.home_win)
    coefs = dict(zip(FEATURE_COLS, m.named_steps["clf"].coef_[0]))
    assert coefs["elo_edge"] > 0
    assert coefs["injury_edge"] > 0
    test = default_test_seasons(dataset)
    assert test == SEASONS[2:]
    preds = walk_forward(dataset, test, C=0.3)
    s = summarize(preds).query("season == 'ALL'").set_index("model")
    assert s.loc["full_model", "log_loss"] < s.loc["home_baseline", "log_loss"]
    assert s.loc["full_model", "accuracy"] > 0.55
    assert not calibration(preds).empty
    assert tune_C(dataset, test) in (0.03, 0.1, 0.3, 1.0, 3.0)


def test_backtest_never_trains_on_test_season(dataset):
    preds = walk_forward(dataset, [2022], C=0.3)
    assert set(preds.season) == {2022}
    # an oracle check: shuffling labels of the TEST season must not change predictions
    d2 = dataset.copy()
    m = d2.season == 2022
    d2.loc[m, "home_win"] = 1 - d2.loc[m, "home_win"]
    preds2 = walk_forward(d2, [2022], C=0.3)
    np.testing.assert_allclose(preds.p_full, preds2.p_full)


# ------------------------------------------------------------------ predict
def test_predict_end_to_end(league, store, dataset):
    _, sched, _, _ = league
    model = make_model(0.3).fit(dataset[FEATURE_COLS], dataset.home_win)
    meta = {"feature_cols": FEATURE_COLS}
    ctx = P.Context(season=2024, store=store, schedules=sched, next_week=None)
    g = sched[(sched.season == 2024) & (sched.week == 10)].iloc[0]
    res = P.predict_matchup(2024, 10, g.home_team, g.away_team, ctx=ctx, model=model, meta=meta)
    assert res["home_win_prob"] + res["away_win_prob"] == pytest.approx(1.0)
    assert res["projected_winner"] in (g.home_team, g.away_team)
    assert "home-field" in res["explanation"]
    assert res["injury_note"].startswith("Injuries")
    with pytest.raises(ValueError):
        P.predict_matchup(2024, 10, g.home_team, g.home_team, ctx=ctx, model=model, meta=meta)
    with pytest.raises(ValueError):
        P.predict_matchup(2024, 10, "XXX", g.away_team, ctx=ctx, model=model, meta=meta)
    # LAR (ESPN style) and LA (nflverse style) must both work
    res2 = P.predict_matchup(2024, 10, "LAR", "KC", ctx=ctx, model=model, meta=meta)
    assert 0 < res2["home_win_prob"] < 1


def test_explanation_direction_matches_who_is_favoured():
    cols = FEATURE_COLS
    raw = np.zeros(len(cols))
    contrib = np.zeros(len(cols))
    i = cols.index("injury_edge")
    raw[i], contrib[i] = 3.0, 0.5   # positive injury_edge == AWAY team more hurt -> favours HOME
    txt = P._explain(cols, raw, contrib, "BUF", "KC", neutral=False)
    assert "KC is missing more" in txt and "favors BUF" in txt
    raw[i], contrib[i] = -3.0, -0.5
    txt = P._explain(cols, raw, contrib, "BUF", "KC", neutral=False)
    assert "BUF is missing more" in txt and "favors KC" in txt
    j = cols.index("def_epa_ewm_edge")
    raw[:], contrib[:] = 0, 0
    raw[j], contrib[j] = -0.2, -0.4
    assert "KC has the edge in recent defensive efficiency" in P._explain(cols, raw, contrib, "BUF", "KC", False)
    assert "neutral-site" in P._explain(cols, raw, contrib, "BUF", "KC", True)


def test_explanation_skips_features_with_wrong_sign_coefficient():
    cols = FEATURE_COLS
    raw = np.zeros(len(cols)); contrib = np.zeros(len(cols))
    j = cols.index("def_epa_ewm_edge")
    raw[j], contrib[j] = 0.3, -0.4      # home is better on raw defense, but coef<0 -> contrib<0
    txt = P._explain(cols, raw, contrib, "BUF", "KC", False)
    assert "defensive" not in txt and "evenly matched" in txt


def test_upcoming_week_and_outdated_model(tmp_path, monkeypatch):
    s = pd.DataFrame({"season": [2025] * 3, "week": [1, 2, 3],
                      "home_score": [10, np.nan, np.nan], "away_score": [3, np.nan, np.nan]})
    assert P.upcoming_week(s, 2025) == 2
    s["home_score"] = 1.0; s["away_score"] = 2.0
    assert P.upcoming_week(s, 2025) is None

    import joblib, json
    monkeypatch.setattr(P, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(P, "META_PATH", tmp_path / "m.json")
    assert P.load_model() == (None, None)
    joblib.dump(object(), P.MODEL_PATH)
    P.META_PATH.write_text(json.dumps({"feature_cols": ["old_feature"]}))
    with pytest.raises(P.ModelOutdatedError):
        P.load_model()


def test_injury_caliber_ignores_game_week_snaps(league):
    """A player ruled OUT plays 0 snaps that week; his caliber must come from earlier weeks only."""
    _, _, inj, snaps = league
    out = inj[(inj.status == "out") & (inj.season == 2022) & (inj.week >= 8)]
    # pick a team-week where an OUT player exists
    r = out.iloc[0]
    from src.features import InjuryModel
    base = InjuryModel(inj, snaps).score(r.season, r.week, r.team)
    s2 = snaps.copy()
    m = (s2.season == r.season) & (s2.week >= r.week)
    s2.loc[m, "snap_pct"] = 1.0
    # add a (tiny, nonzero) game-week snap row for the OUT player himself
    s2 = pd.concat([s2, pd.DataFrame([dict(season=r.season, week=r.week, team=r.team,
                                           player_name=r.player_name, snap_pct=1.0)])], ignore_index=True)
    assert InjuryModel(inj, s2).score(r.season, r.week, r.team) == pytest.approx(base)
    # ...but earlier weeks DO matter (starter vs. backup changes the weight)
    s3 = snaps.copy()
    m = (s3.season == r.season) & (s3.week < r.week) & (s3.week >= r.week - 6)
    s3.loc[m, "snap_pct"] = 100.0
    assert InjuryModel(inj, s3).score(r.season, r.week, r.team) != pytest.approx(base)
