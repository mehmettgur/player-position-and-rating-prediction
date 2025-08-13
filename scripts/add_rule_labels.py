import argparse
import pandas as pd
from soccer_rating.labeling.rule_based import rule_general, RULE_MAP_GENERAL, rule_detailed
from soccer_rating.data.io import read_parquet, to_parquet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_in', required=True, help='Parquet input (from merge_player_data)')
    ap.add_argument('--data_out', required=True, help='Parquet output with rule labels')
    args = ap.parse_args()

    df = read_parquet(args.data_in).copy()

    # Genel (4 rol) rule-based
    df['rule_position_code'] = df.apply(rule_general, axis=1)
    df['rule_position_label'] = df['rule_position_code'].map(RULE_MAP_GENERAL)

    # Detaylı (7 rol) rule-based
    df['rule_detailed_position'] = df.apply(rule_detailed, axis=1)

    to_parquet(df, args.data_out)
    print(f"[OK] Rule-based labels added → {args.data_out}")

if __name__ == '__main__':
    main()
