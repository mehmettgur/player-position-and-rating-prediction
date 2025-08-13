import argparse
import pandas as pd
from soccer_rating.labeling.kmeans_label import general_by_Y, detailed_with_side
from soccer_rating.data.io import read_parquet, to_parquet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--positions', required=True, help='Input positions parquet')
    ap.add_argument('--out', required=True, help='Output parquet (with labels)')
    args = ap.parse_args()

    pos = read_parquet(args.positions)
    pos = general_by_Y(pos)
    pos = detailed_with_side(pos)
    to_parquet(pos, args.out)
    print(f"[OK] Labeled positions saved to {args.out}")

if __name__ == '__main__':
    main()
