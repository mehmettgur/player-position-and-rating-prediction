import argparse
import pandas as pd
from soccer_rating.data.sqlite_loader import connect, read_player_attributes, latest_attributes
from soccer_rating.data.io import read_parquet, to_parquet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sqlite', required=True)
    ap.add_argument('--positions', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    conn = connect(args.sqlite)
    attrs = read_player_attributes(conn)
    latest = latest_attributes(attrs)

    pos = read_parquet(args.positions).rename(columns={'player_api_id':'player_api_id'})
    merged = pos.merge(latest, on='player_api_id', how='inner')
    # clean + helper columns similar to original
    merged = merged.dropna()
    print(f"[INFO] merged shape: {merged.shape}")
    to_parquet(merged, args.out)
    print(f"[OK] Player data saved to {args.out}")

if __name__ == '__main__':
    main()
