import pandas as pd
from sklearn.cluster import KMeans

def general_by_Y(position_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    df = position_df.copy()
    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(df[['avg_Y']].values)
    df['cluster_y'] = km.labels_
    # derive thresholds and map to labels by center order
    centers = pd.DataFrame(km.cluster_centers_, columns=['avg_Y'])
    centers['cluster'] = centers.index
    centers = centers.sort_values('avg_Y').reset_index(drop=True)
    label_map = {
        centers.loc[0,'cluster']: 'GK',
        centers.loc[1,'cluster']: 'Defense',
        centers.loc[2,'cluster']: 'Midfield',
        centers.loc[3,'cluster']: 'Forward',
    }
    df['kmeans_position_label'] = df['cluster_y'].map(label_map)
    return df

def detailed_with_side(df_with_general: pd.DataFrame) -> pd.DataFrame:
    df = df_with_general.copy()
    df['kmeans_detailed_position'] = df.apply(
        lambda r: 'GK' if r['kmeans_position_label']=='GK' else f"{r['side']} {r['kmeans_position_label']}",
        axis=1
    )
    return df
