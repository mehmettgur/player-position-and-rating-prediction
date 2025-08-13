import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Union

def _finalize(fig: plt.Figure, save_path: Optional[str], show: bool) -> plt.Figure:
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def scatter_positions(
    position_df: pd.DataFrame,
    x: str = "avg_X",
    y: str = "avg_Y",
    label_col: str = "kmeans_position_label",
    title: str = "Player Positions",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = [l for l in sorted(position_df[label_col].dropna().unique())]
    for label in labels:
        s = position_df[position_df[label_col] == label]
        ax.scatter(s[x], s[y], s=10, alpha=0.6, label=str(label))
    ax.set_title(title)
    ax.set_xlabel("Average X")
    ax.set_ylabel("Average Y")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return _finalize(fig, save_path, show)

def bar_model_summary(
    results_df: pd.DataFrame,
    title: str = "Model Comparison: Average RMSE & R²",
    group_col: str = "BaseModel",
    rmse_col: str = "RMSE",
    r2_col: str = "R2",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    summary = results_df.groupby(group_col).agg({rmse_col: "mean", r2_col: "mean"}).reset_index()
    x = np.arange(len(summary))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, summary[rmse_col], width=w, label="RMSE")
    ax.bar(x + w / 2, summary[r2_col], width=w, label="R²")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x, summary[group_col])
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    return _finalize(fig, save_path, show)

def feature_importance_heatmap(
    fi_source: Union[str, pd.DataFrame],
    model_id: str = "M6-Ridge",
    top_n: int = 5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    if isinstance(fi_source, str):
        df = pd.read_csv(fi_source)
    else:
        df = fi_source.copy()

    if "Model_ID" not in df.columns or "Position" not in df.columns:
        raise ValueError("Input must contain columns: Model_ID, Position, Feature, Importance")

    sub = df[(df["Model_ID"] == model_id) & (df["Importance"] > 0)].copy()
    if sub.empty:
        raise ValueError(f"No positive importances for model_id '{model_id}'")

    top_rows = (
        sub.sort_values(["Position", "Importance"], ascending=[True, False])
           .groupby("Position")
           .head(top_n)
    )

    pivot = top_rows.pivot_table(index="Feature", columns="Position", values="Importance", aggfunc="max")
    desired_cols = [c for c in ["GK", "Defense", "Midfield", "Forward"] if c in pivot.columns]
    pivot = pivot[desired_cols] if desired_cols else pivot
    data = pivot.fillna(0.0).values

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(data, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Coefficient Value (Impact on Rating)")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ttl = title or f"Top {top_n} Critical Skills by Position (Model: {model_id})"
    ax.set_title(ttl, fontweight="bold")
    ax.set_xlabel("Position")
    ax.set_ylabel("Player Skill")
    fig.tight_layout()
    return _finalize(fig, save_path, show)
