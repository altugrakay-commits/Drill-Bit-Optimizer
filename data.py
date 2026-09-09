# data.py
# Data loading, generation, and preprocessing.

import numpy as np
import pandas as pd
from config import DATA_DIR, metallurgy_features, geology_features

def generate_synthetic_data(n_samples=50):
    """Create synthetic metallurgy and geology data for testing."""
    metal_data = {feat: np.random.uniform(0.2, 1.2 if 'Mean' in feat else 600.0, n_samples)
                  for feat in metallurgy_features}
    metal_df = pd.DataFrame(metal_data)
    metal_df['Grade'] = 'G30'
    metal_df['Cast'] = 'C1'
    metal_df['Variant'] = 'V2'

    geo_data = {feat: np.random.uniform(0.01, 70.0 if 'SiO2' in feat else 4.0, n_samples)
                for feat in geology_features}
    geo_df = pd.DataFrame(geo_data)
    geo_df['Rock_Name'] = 'Granite'
    geo_df['Rock_ID'] = 'R101'

    return metal_df, geo_df

def load_and_prepare_data(metal_path=None, geo_path=None):
    """Load data from CSV or generate synthetic; cross‑join into training matrix."""
    if metal_path and geo_path:
        metal_df = pd.read_csv(metal_path, sep=";")
        geo_df = pd.read_csv(geo_path, sep=";")
    else:
        metal_df, geo_df = generate_synthetic_data()
    # Cross‑join (cartesian product)
    cross_df = metal_df.assign(key=1).merge(geo_df.assign(key=1), on='key').drop('key', axis=1)
    cross_df['Traceability_Map'] = cross_df['Grade'].astype(str) + "_" + cross_df['Rock_ID'].astype(str)
    return cross_df, metal_df, geo_df

def compute_stats(df, feature_cols):
    """Compute mean and standard deviation for each feature."""
    stats = {}
    for col in feature_cols:
        mu = float(df[col].mean())
        sigma = float(df[col].std())
        stats[col] = {'mu': mu, 'sig': sigma if sigma > 1e-6 else 1.0}
    return stats

def dual_track_fill(df_tensor):
    """Extract metallurgy and geology feature matrices from cross‑joined dataframe."""
    X_metal = df_tensor[metallurgy_features].to_numpy()
    X_geo = df_tensor[geology_features].to_numpy()
    # metadata placeholder (not used)
    return X_metal, X_geo, None

def normalize_data(df, stats, feature_cols):
    """Apply Z‑score normalisation using pre‑computed statistics."""
    df_norm = df.copy()
    for col in feature_cols:
        mu = stats[col]['mu']
        sigma = stats[col]['sig']
        df_norm[col] = (df_norm[col] - mu) / sigma
    return df_norm