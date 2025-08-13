import pandas as pd

def extract_position_long(match_df: pd.DataFrame) -> pd.DataFrame:
    # Build long format from home/away X/Y
    home_rows, away_rows = [], []
    for _, row in match_df.iterrows():
        match_id = row['id']
        for i in range(1,12):
            pid_h = row.get(f'home_player_{i}')
            x_h = row.get(f'home_player_X{i}')
            y_h = row.get(f'home_player_Y{i}')
            if pd.notna(pid_h) and pd.notna(x_h) and pd.notna(y_h):
                home_rows.append({'match_id': match_id, 'player_api_id': int(pid_h), 'X': float(x_h), 'Y': float(y_h), 'team': 'home'})

            pid_a = row.get(f'away_player_{i}')
            x_a = row.get(f'away_player_X{i}')
            y_a = row.get(f'away_player_Y{i}')
            if pd.notna(pid_a) and pd.notna(x_a) and pd.notna(y_a):
                away_rows.append({'match_id': match_id, 'player_api_id': int(pid_a), 'X': float(x_a), 'Y': float(y_a), 'team': 'away'})

    return pd.DataFrame(home_rows + away_rows)

def average_positions(position_long: pd.DataFrame) -> pd.DataFrame:
    g = position_long.groupby('player_api_id').agg({'X':'mean','Y':'mean'}).reset_index()
    g.rename(columns={'X':'avg_X','Y':'avg_Y'}, inplace=True)
    return g

def add_side_column(position_df: pd.DataFrame) -> pd.DataFrame:
    df = position_df.copy()
    df['side'] = df['avg_X'].apply(lambda x: 'Center' if 3.5 < x < 6.5 else 'Side')
    return df
