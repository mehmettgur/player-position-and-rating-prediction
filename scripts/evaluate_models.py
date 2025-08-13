import argparse, json
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', required=True)
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    all_rows = []
    for csv_path in runs.rglob('results.csv'):
        df = pd.read_csv(csv_path)
        df['Run'] = csv_path.parent.name
        all_rows.append(df)
    if not all_rows:
        print("[WARN] No results.csv found under runs_dir")
        return
    all_df = pd.concat(all_rows, ignore_index=True)

    # averages per Model
    avg = (all_df.groupby('Model')
                  .apply(lambda x: pd.Series({
                      'RMSE': (x['RMSE']*x['Samples']).sum()/x['Samples'].sum(),
                      'R2': (x['R2']*x['Samples']).sum()/x['Samples'].sum()
                  }))
                  .reset_index())
    print(avg.sort_values('RMSE').to_string(index=False))

if __name__ == '__main__':
    main()
