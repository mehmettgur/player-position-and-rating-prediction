import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from soccer_rating.modeling.trainers import _base_features
from soccer_rating.config.labels import RULE_FEATURES_GENERAL, RULE_FEATURES_DETAILED

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--runs_dir', required=False, help='optional')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='M6', help='base model id to analyze (e.g., M6)')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    all_numeric = _base_features(df)

    label_col = 'rule_position_label' if args.model in ['M5','M6'] else (
        'kmeans_position_label' if args.model in ['M1','M2'] else (
            'kmeans_detailed_position' if args.model in ['M3','M4'] else 'rule_detailed_position'
        )
    )
    use_all = args.model in ['M2','M4','M6','M8']
    is_detailed = 'detailed' in label_col

    reg = Ridge()  # you can swap to RF/XGB/EN here

    rows = []
    for pos in sorted(df[label_col].dropna().unique()):
        subset = df[df[label_col]==pos]
        features = all_numeric if use_all else (RULE_FEATURES_DETAILED[pos] if is_detailed else RULE_FEATURES_GENERAL.get(pos))
        if not features: 
            continue
        X = subset[features].dropna()
        y = subset.loc[X.index, 'overall_rating']
        if len(X) < 10:
            continue
        reg.fit(X, y)
        importances = getattr(reg, 'coef_', None)
        if importances is None:
            continue
        for fname, weight in zip(X.columns, importances):
            rows.append({'Model': args.model, 'Position': pos, 'Feature': fname, 'Importance': float(weight)})

    fi = pd.DataFrame(rows)
    fi.to_csv(out_dir / f'feature_importance_{args.model}.csv', index=False)
    print(f"[OK] Saved to {out_dir / f'feature_importance_{args.model}.csv'}")

if __name__ == '__main__':
    main()
