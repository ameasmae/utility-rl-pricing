# shap_analysis.py

"""
SHAP analysis for:
1) Utility-only model (features: Mileage, Profit_Margin, Time_Diff)
2) Utility + Attention model (features + S_att)

We fit a tree-based regressor to approximate Utility and use SHAP.TreeExplainer
for global feature importance plots.

Outputs:
- shap_importance_utility_only.csv
- shap_importance_utility_attention.csv
- shap_bar_utility_only.png
- shap_bar_utility_attention.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from utility_and_markov import (
    label_regular_irregular,
    estimate_markov_transition,
    steady_state_probabilities,
    compute_penalty_rate,
    compute_utility,
)
from attention_model import (
    train_attention_model,
    compute_attention_weighted_states,
    FEATURE_COLS,
)


def prepare_data(csv_name: str = "synthetic_vehicle_maintenance.csv"):
    here = Path(__file__).resolve().parent
    data_path = here / csv_name

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    if "Service_Date" in df.columns:
        df["Service_Date"] = pd.to_datetime(df["Service_Date"])

    # Regular flag
    df = label_regular_irregular(
        df,
        max_mileage_diff=10_000,
        max_time_years=1.0,
    )

    # Markov, penalty, utility
    P = estimate_markov_transition(df)
    pi1, pi2 = steady_state_probabilities(P)
    rho = compute_penalty_rate(df, r=0.1, T=8)

    beta = 0.5
    df = compute_utility(
        df,
        beta=beta,
        pi1=pi1,
        pi2=pi2,
        rho=rho,
    )

    # Attention
    att_weights = train_attention_model(df)
    df_att = compute_attention_weighted_states(
        df,
        attention_weights=att_weights,
        feature_cols=FEATURE_COLS,
    )

    # Build S_att = weighted sum of raw features (for attention model)
    # FEATURE_COLS and att_weights must align in order
    X_raw = df[FEATURE_COLS].values
    s_att = (X_raw * att_weights.reshape(1, -1)).sum(axis=1)
    df_att["S_att"] = s_att

    return df, df_att, att_weights


def train_tree_and_shap(df, feature_cols, target_col="Utility"):
    """
    Fit a RandomForestRegressor and compute global SHAP importance
    (mean(|SHAP|) per feature).
    """
    X = df[feature_cols].values
    y = df[target_col].values

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    # For speed, sample up to 2000 points
    if len(df) > 2000:
        df_sample = df.sample(2000, random_state=42)
    else:
        df_sample = df

    X_sample = df_sample[feature_cols].values
    shap_values = explainer.shap_values(X_sample)

    # shap_values has shape (n_samples, n_features)
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return importance_df


def plot_shap_bar(importance_df: pd.DataFrame, title: str, out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.bar(
        importance_df["feature"],
        importance_df["mean_abs_shap"],
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    here = Path(__file__).resolve().parent
    df, df_att, att_weights = prepare_data()

    # ---------- Utility-only model ----------
    feature_cols_base = ["Mileage", "Profit_Margin", "Time_Diff"]
    print("\n=== SHAP: Utility-only model ===")
    shap_base = train_tree_and_shap(
        df,
        feature_cols=feature_cols_base,
        target_col="Utility",
    )
    print(shap_base)

    csv_base = here / "shap_importance_utility_only.csv"
    shap_base.to_csv(csv_base, index=False)
    print(f"Saved utility-only SHAP importance to: {csv_base}")

    png_base = here / "shap_bar_utility_only.png"
    plot_shap_bar(
        shap_base,
        title="Utility-only model: global SHAP importance",
        out_path=png_base,
    )
    print(f"Saved utility-only SHAP bar plot to: {png_base}")

    # ---------- Utility + Attention model ----------
    print("\n=== SHAP: Utility + Attention model ===")
    feature_cols_att = ["Mileage", "Profit_Margin", "Time_Diff", "S_att"]
    shap_att = train_tree_and_shap(
        df_att,
        feature_cols=feature_cols_att,
        target_col="Utility",
    )
    print(shap_att)

    csv_att = here / "shap_importance_utility_attention.csv"
    shap_att.to_csv(csv_att, index=False)
    print(f"Saved utility+attention SHAP importance to: {csv_att}")

    png_att = here / "shap_bar_utility_attention.png"
    plot_shap_bar(
        shap_att,
        title="Utility + Attention model: global SHAP importance",
        out_path=png_att,
    )
    print(f"Saved utility+attention SHAP bar plot to: {png_att}")


if __name__ == "__main__":
    main()
