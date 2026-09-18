import pandas as pd

def load_and_preprocess_data(seasons=['2223', '2324', '2425', '2526', '2627']):
    df_list = []
    for season in seasons:
        url = f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv'
        try:
            temp_df = pd.read_csv(url)
            df_list.append(temp_df)
            print(f"Successfully loaded season {season}")
        except Exception as e:
            print(f"Failed to load season {season}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    # Perspective frames
    h_df = pd.DataFrame({
        'Match_ID': df.index, 'Date': df['Date'], 'Team': df['HomeTeam'],
        'GF': df['FTHG'], 'GA': df['FTAG'], 'ST': df['HST'],
        'Pts': df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    })

    a_df = pd.DataFrame({
        'Match_ID': df.index, 'Date': df['Date'], 'Team': df['AwayTeam'],
        'GF': df['FTAG'], 'GA': df['FTHG'], 'ST': df['AST'],
        'Pts': df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    })

    timeline = pd.concat([h_df, a_df]).sort_values(['Team', 'Date']).reset_index(drop=True)

    stats = ['GF', 'GA', 'ST', 'Pts']
    roll_names = [f"Roll_{s}" for s in stats]
    timeline[roll_names] = (
        timeline.groupby('Team')[stats]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    home_roll = timeline[timeline['Match_ID'].isin(df.index)][['Match_ID', 'Team'] + roll_names]
    home_roll.columns = ['Match_ID', 'HomeTeam'] + [f"Home_{c}" for c in roll_names]

    away_roll = timeline[timeline['Match_ID'].isin(df.index)][['Match_ID', 'Team'] + roll_names]
    away_roll.columns = ['Match_ID', 'AwayTeam'] + [f"Away_{c}" for c in roll_names]

    home_roll = home_roll.drop_duplicates(subset=['Match_ID', 'HomeTeam'])
    away_roll = away_roll.drop_duplicates(subset=['Match_ID', 'AwayTeam'])

    features_df = df[['HomeTeam', 'AwayTeam', 'FTR']].copy()
    features_df['Match_ID'] = df.index
    features_df = features_df.merge(home_roll, on=['Match_ID', 'HomeTeam'])
    features_df = features_df.merge(away_roll, on=['Match_ID', 'AwayTeam'])
    features_df = features_df.fillna(0)

    features_df['Target'] = features_df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
    num_features = [col for col in features_df.columns if 'Roll_' in col]

    return features_df, num_features, home_roll, away_roll