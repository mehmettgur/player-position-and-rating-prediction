import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def scatter_positions(position_df: pd.DataFrame, x='avg_X', y='avg_Y', label_col='kmeans_position_label', title='Player Positions'):
    plt.figure(figsize=(7,7))
    for label in sorted(position_df[label_col].dropna().unique()):
        s = position_df[position_df[label_col]==label]
        plt.scatter(s[x], s[y], s=10, alpha=0.6, label=str(label))
    plt.title(title)
    plt.xlabel('Average X')
    plt.ylabel('Average Y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def bar_model_summary(summary_df: pd.DataFrame, title='Average RMSE & R2'):
    fig = plt.figure(figsize=(10,5))
    x = np.arange(len(summary_df))
    width = 0.35
    plt.bar(x - width/2, summary_df['RMSE'], width, label='RMSE')
    plt.bar(x + width/2, summary_df['R2'], width, label='R2')
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(x, summary_df['Model'], rotation=0)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    return fig
