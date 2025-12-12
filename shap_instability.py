"""
shap_instability.py

Compute a SHAP instability index (mean std of global mean|SHAP| across runs)
for:
  (A) Utility-only features: [Mileage, Profit_Margin, Time_Diff]
  (B) Utility + Attention features: [Mileage, Profit_Margin, Time_Diff] + att_* columns

Robust to NaNs via median imputation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from utility_and_markov import (
    label_regular_irregular,
    estimate_markov_transition,
    steady_state_probabilities,
    compute_penalty_rate,
    compute_utility,
)
from attention_model import train_attention_model, compute_attention_weighted_states

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import shap


def _prepare_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Service_Date"] = pd.to_datetime(df["Service_Date"])

    df = label_regular_irregular(df, max_mileage_diff=10_000, max_time_years=1.0)

    P = estimate_markov_transition(df)
    pi1, pi2 = steady_state_probabilities(P)
    rho = compute_penalty_rate(df, r=0.1, T=8)

    df = compute_utility(df, beta=0.5, pi1=pi1, pi2=pi2, rho=rho)
    return df


def _fit_and_shap_mean_abs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    background_size: int = 256,
    seed: int = 0,
) -> pd.Series:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)

    rng = np.random.default_rng(seed)
    if len(X_train) > background_size:
        bg_idx = rng.choice(len(X_train), size=background_size, replace=False)
        X_bg = X_train.iloc[bg_idx]
    else:
        X_bg = X_train

    def f(x_np):
        x_df = pd.DataFrame(x_np, columns=feature_names)
        return model.predict(x_df)

    explainer = shap.KernelExplainer(f, X_bg)
    shap_vals = explainer.shap_values(X_explain, nsamples=200)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    return pd.Series(mean_abs, index=feature_names)


def main():
    here = Path(__file__).resolve().parent
    data_path = here / "synthetic_vehicle_maintenance.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = _prepare_df(data_path)

    # ---- Attention weights + attention-weighted columns att_* ----
    att_w = train_attention_model(df)  # returns np.ndarray weights
    df_att = compute_attention_weighted_states(df, attention_weights=att_w)

    # Confirm expected columns exist
    needed_att_cols = ["att_Mileage", "att_Profit_Margin", "att_Time_Diff"]
    missing = [c for c in needed_att_cols if c not in df_att.columns]
    if missing:
        raise KeyError(
            f"Missing attention columns {missing}. "
            f"df_att columns are: {list(df_att.columns)}"
        )

    y = df["Utility"].values.astype(float)

    feat_base = ["Mileage", "Profit_Margin", "Time_Diff"]
    feat_att = ["Mileage", "Profit_Margin", "Time_Diff", "att_Mileage", "att_Profit_Margin", "att_Time_Diff"]

    n_runs = 20
    explain_n = 800
    train_n = 4000
    rng = np.random.default_rng(42)

    base_runs = []
    att_runs = []

    for r in range(n_runs):
        idx_train = rng.choice(len(df), size=min(train_n, len(df)), replace=False)
        idx_expl = rng.choice(len(df), size=min(explain_n, len(df)), replace=False)

        # IMPORTANT: use iloc because idx_* are positional indices
        Xtr_base = df.iloc[idx_train][feat_base]
        Xex_base = df.iloc[idx_expl][feat_base]

        Xtr_att = df_att.iloc[idx_train][feat_att]
        Xex_att = df_att.iloc[idx_expl][feat_att]

        ytr = y[idx_train]

        mean_abs_base = _fit_and_shap_mean_abs(
            X_train=Xtr_base,
            y_train=ytr,
            X_explain=Xex_base,
            feature_names=feat_base,
            seed=r,
        )
        base_runs.append(mean_abs_base)

        mean_abs_att = _fit_and_shap_mean_abs(
            X_train=Xtr_att,
            y_train=ytr,
            X_explain=Xex_att,
            feature_names=feat_att,
            seed=100 + r,
        )
        att_runs.append(mean_abs_att)

        print(f"Run {r+1:02d}/{n_runs} done.")

    base_mat = pd.DataFrame(base_runs)
    att_mat = pd.DataFrame(att_runs)

    base_instability = float(base_mat.std(axis=0).mean())
    att_instability = float(att_mat.std(axis=0).mean())

    summary = pd.DataFrame(
        [
            {"Model": "Utility-only", "SHAP_Instability": base_instability},
            {"Model": "Utility + Attention", "SHAP_Instability": att_instability},
        ]
    )
    summary["Stability_Gain_x"] = summary["SHAP_Instability"].max() / summary["SHAP_Instability"]

    out_summary = here / "shap_instability_summary.csv"
    out_base = here / "shap_meanabs_runs_utility_only.csv"
    out_att = here / "shap_meanabs_runs_utility_attention.csv"

    summary.to_csv(out_summary, index=False)
    base_mat.to_csv(out_base, index=False)
    att_mat.to_csv(out_att, index=False)

    print("\n=== SHAP instability results ===")
    print(summary)
    print(f"\nSaved: {out_summary}")
    print(f"Saved: {out_base}")
    print(f"Saved: {out_att}")


if __name__ == "__main__":
    main()
