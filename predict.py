import os
import joblib
import pandas as pd

def get_prediction(home_team, away_team):
    required_files = ['model.joblib', 'columns.joblib', 'num_features.joblib', 'home_roll.joblib', 'away_roll.joblib']
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} missing. Run 'python train.py' first.")

    model = joblib.load('model.joblib')
    feature_columns = joblib.load('columns.joblib')
    num_features = joblib.load('num_features.joblib')
    home_roll = joblib.load('home_roll.joblib')
    away_roll = joblib.load('away_roll.joblib')

    home_col = f"HomeTeam_{home_team}"
    away_col = f"AwayTeam_{away_team}"

    if home_col not in feature_columns:
        raise ValueError(f"'{home_team}' was not found in the training dataset.")
    if away_col not in feature_columns:
        raise ValueError(f"'{away_team}' was not found in the training dataset.")

    match_data = {col: 0.0 for col in feature_columns}
    match_data[home_col] = 1.0
    match_data[away_col] = 1.0

    recent_home = home_roll[home_roll['HomeTeam'] == home_team]
    recent_away = away_roll[away_roll['AwayTeam'] == away_team]

    if recent_home.empty or recent_away.empty:
        raise ValueError("Missing form history for one of the specified teams.")

    latest_home_form = recent_home.iloc[-1]
    latest_away_form = recent_away.iloc[-1]

    for col in num_features:
        if col.startswith('Home_') and col in latest_home_form:
            match_data[col] = float(latest_home_form[col])
        elif col.startswith('Away_') and col in latest_away_form:
            match_data[col] = float(latest_away_form[col])

    match_df = pd.DataFrame([match_data])[feature_columns]

    pred_code = model.predict(match_df)[0]
    probabilities = model.predict_proba(match_df)[0]
    label_map = {0: f"{away_team} Win", 1: "Draw", 2: f"{home_team} Win"}

    return {
        "prediction": label_map[pred_code],
        "probabilities": {
            f"{away_team} Win": f"{probabilities[0] * 100:.1f}%",
            "Draw": f"{probabilities[1] * 100:.1f}%",
            f"{home_team} Win": f"{probabilities[2] * 100:.1f}%",
        }
    }