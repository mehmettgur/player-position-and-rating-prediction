import argparse, json
import pandas as pd
from soccer_rating.modeling.trainers import run_all_models
from soccer_rating.data.io import read_parquet
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True, help='Output directory for run')
    args = ap.parse_args()

    df = read_parquet(args.data)
    results = run_all_models(df)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / 'results.csv', index=False)
    print(f"[OK] Saved results to {out_dir/'results.csv'}")

if __name__ == '__main__':
    main()
