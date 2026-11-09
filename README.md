# NFL Matchup Predictor

Win probability for an NFL game from pre-game information only.

## Run

    pip install -r requirements.txt
    python -m src.train_model        # downloads data, backtests, saves models/model.joblib + meta.json
    streamlit run app.py
    pytest                           # offline tests on synthetic data (incl. leakage checks)

Training uses seasons 2019..current (rolls forward automatically). Options:
`--seasons 2020 2021 ...`, `--C 0.3`, `--skip_backtest`.
`python -m src.backtest` re-runs the walk-forward evaluation from `models/training_rows.csv`.

## Layout

    app.py               Streamlit UI (data and model cached)
    src/teams.py         team codes + canonicalisation (LA/LAR, WSH/WAS, ...)
    src/data.py          ALL downloads (nfl_data_py, ESPN live injuries)
    src/features.py      pre-game features: EPA form, turnovers, Elo, injuries
    src/dataset.py       builds one row per completed game
    src/backtest.py      walk-forward evaluation vs baselines, C tuning
    src/train_model.py   training entry point
    src/predict.py       inference + explanation
    tests/               synthetic-league tests (leakage, Elo, backtest, explanations)

## Features (all "edges": positive = better for the home team)

off_epa_ewm_edge, def_epa_ewm_edge, net_epa_season_edge, to_margin_edge,
elo_edge, injury_edge, neutral_site.
