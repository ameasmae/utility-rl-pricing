# attention_model.py

import numpy as np
import pandas as pd
from typing import List

# Features used in the attention mechanism (consistent with the paper)
FEATURE_COLS = ["Mileage", "Profit_Margin", "Time_Diff"]


def minmax_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Min-max scale selected columns to [0, 1].
    """
    df = df.copy()
    for c in cols:
        cmin = df[c].min()
        cmax = df[c].max()
        if cmax > cmin:
            df[c] = (df[c] - cmin) / (cmax - cmin)
        else:
            df[c] = 0.0
    return df


def train_attention_model(df: pd.DataFrame) -> np.ndarray:
    """
    Lightweight attention mechanism:

    1. Min-max scale the feature columns.
    2. Compute absolute correlation between each feature and the Utility column.
    3. Normalize these correlations to obtain attention weights that sum to 1.

    Returns:
        attention_weights: np.ndarray of shape (len(FEATURE_COLS),)
    """
    df_scaled = minmax_scale(df, FEATURE_COLS)

    # Use the utility column from the RL formulation
    if "Utility" in df.columns:
        U = df["Utility"].values
    elif "U_utility" in df.columns:
        # Fallback if you ever kept the old naming
        U = df["U_utility"].values
    else:
        raise KeyError("Expected a 'Utility' (or 'U_utility') column in the dataframe.")

    att_raw = []
    for c in FEATURE_COLS:
        x = df_scaled[c].values
        if np.std(x) == 0 or np.std(U) == 0:
            att_raw.append(0.0)
        else:
            corr = np.corrcoef(x, U)[0, 1]
            att_raw.append(abs(corr))

    att_raw = np.array(att_raw, dtype=float)
    if att_raw.sum() == 0:
        attention_weights = np.ones_like(att_raw) / len(att_raw)
    else:
        attention_weights = att_raw / att_raw.sum()

    return attention_weights


def compute_attention_weighted_states(
    df: pd.DataFrame,
    attention_weights: np.ndarray,
    feature_cols: List[str] = None,
    out_prefix: str = "att_",
) -> pd.DataFrame:
    """
    Compute attention-weighted feature columns.

    For each feature x_j in feature_cols and its attention weight w_j,
    add a new column:   out_prefix + feature_name = w_j * x_j

    Example:
        FEATURE_COLS = ["Mileage", "Profit_Margin", "Time_Diff"]
        → columns: "att_Mileage", "att_Profit_Margin", "att_Time_Diff"
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if len(attention_weights) != len(feature_cols):
        raise ValueError(
            f"Length of attention_weights ({len(attention_weights)}) "
            f"does not match number of feature_cols ({len(feature_cols)})."
        )

    df = df.copy()
    for w, col in zip(attention_weights, feature_cols):
        df[f"{out_prefix}{col}"] = df[col] * w

    return df
