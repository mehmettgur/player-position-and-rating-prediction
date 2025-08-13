import argparse, os
import pandas as pd
from soccer_rating.data.sqlite_loader import connect, read_match
from soccer_rating.features.positions import extract_position_long, average_positions, add_side_column
from soccer_rating.data.io import to_parquet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sqlite', required=True, help='Path to database.sqlite')
    ap.add_argument('--out', required=True, help='Output parquet path')
    args = ap.parse_args()

    conn = connect(args.sqlite)
    match_df = read_match(conn)
    pos_long = extract_position_long(match_df)
    pos_avg = average_positions(pos_long)
    pos_avg = add_side_column(pos_avg)
    to_parquet(pos_avg, args.out)
    print(f"[OK] Positions saved to {args.out}")

if __name__ == '__main__':
    main()
