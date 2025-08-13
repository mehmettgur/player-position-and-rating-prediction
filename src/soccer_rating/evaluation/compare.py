import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..config.labels import GENERAL_ORDER, DETAILED_ORDER

def accuracy_simple(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    c = (df[col_a] == df[col_b]).mean()
    return float(c)

def confusion_general(df: pd.DataFrame, true_col: str, pred_col: str) -> np.ndarray:
    return confusion_matrix(df[true_col], df[pred_col], labels=GENERAL_ORDER)

def confusion_detailed(df: pd.DataFrame, true_col: str, pred_col: str) -> np.ndarray:
    return confusion_matrix(df[true_col], df[pred_col], labels=DETAILED_ORDER)
