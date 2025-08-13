import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from ..config.labels import RULE_FEATURES_GENERAL, RULE_FEATURES_DETAILED
from .frameworks import MODEL_PARAMS

def _base_features(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include='number').columns.tolist()
    exclude = {'player_api_id','player_fifa_api_id','overall_rating','avg_X','avg_Y','cluster_y','rule_position_code'}
    return [c for c in numeric if c not in exclude]

def _feature_map_for_label(pos: str, detailed: bool) -> List[str] | None:
    m = RULE_FEATURES_DETAILED if detailed else RULE_FEATURES_GENERAL
    return m.get(pos)

def _fit_and_eval(model, X_train, X_test, y_train, y_test) -> Tuple[float, float]:
    from sklearn.metrics import root_mean_squared_error, r2_score
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return float(root_mean_squared_error(y_test, y_pred)), float(r2_score(y_test, y_pred))

def run_all_models(player_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    results = []
    all_numeric = _base_features(player_df)

    regressors = {
        'Ridge': Ridge(),
        'RF': RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        'XGB': xgb.XGBRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        'EN': ElasticNet(random_state=random_state)
    }

    for model_id, cfg in MODEL_PARAMS.items():
        label_col = cfg.label_col
        is_detailed = 'detailed' in label_col

        for pos in sorted(player_df[label_col].dropna().unique()):
            subset = player_df[player_df[label_col] == pos]
            if subset.empty:
                continue

            features = all_numeric if cfg.use_all_features else _feature_map_for_label(pos, detailed=is_detailed)
            if not features:
                continue

            X = subset[features].dropna()
            y = subset.loc[X.index, 'overall_rating']

            if len(X) < 10:
                continue

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state)
            for reg_name, reg in regressors.items():
                rmse, r2 = _fit_and_eval(reg, Xtr, Xte, ytr, yte)
                results.append({
                    'Model': f"{model_id}-{reg_name}",
                    'BaseModel': model_id,
                    'Regressor': reg_name,
                    'Position': pos,
                    'Samples': int(len(X)),
                    'RMSE': round(rmse, 4),
                    'R2': round(r2, 4)
                })

    return pd.DataFrame(results).sort_values(['Model','Position']).reset_index(drop=True)
