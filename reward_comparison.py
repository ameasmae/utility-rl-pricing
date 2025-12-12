# reward_comparison.py

"""
Reward comparison for:
1) Traditional RL (margin-only reward)
2) Utility-based RL
3) Utility + Attention

Uses the same data pipeline as main_experiment.py:
- synthetic_vehicle_maintenance.csv
- utility_and_markov (Regular_Flag, Utility)
- attention_model (att_Mileage, att_Profit_Margin, att_Time_Diff)
- q_learning_env (PricingEnv, run_q_learning)

Outputs:
- Printed summary of mean/std rewards (last 200 episodes)
- reward_comparison_results.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

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
from q_learning_env import PricingEnv, run_q_learning


def prepare_data(csv_name: str = "synthetic_vehicle_maintenance.csv"):
    """Load CSV, compute Regular_Flag, Utility, and attention-weighted features."""

    here = Path(__file__).resolve().parent
    data_path = here / csv_name

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Ensure date is in datetime (useful for Markov)
    if "Service_Date" in df.columns:
        df["Service_Date"] = pd.to_datetime(df["Service_Date"])

    # 1) Regular / irregular flags
    df = label_regular_irregular(
        df,
        max_mileage_diff=10_000,
        max_time_years=1.0,
    )

    # 2) Markov + penalty rate + steady-state
    P = estimate_markov_transition(df)
    pi1, pi2 = steady_state_probabilities(P)
    rho = compute_penalty_rate(df, r=0.1, T=8)

    print("=== Markov retention and penalty parameters ===")
    print(f"Transition matrix P:\n{P}")
    print(f"Steady-state probabilities: π1={pi1:.4f}, π2={pi2:.4f}")
    print(f"Penalty rate ρ = {rho:.4f}")

    # 3) Utility
    beta = 0.5
    df = compute_utility(
        df,
        beta=beta,
        pi1=pi1,
        pi2=pi2,
        rho=rho,
    )  # adds Immediate_U, Future_U, Utility

    # 4) Attention model and attention-weighted states
    print("\n=== Training attention model ===")
    att_weights = train_attention_model(df)
    df_att = compute_attention_weighted_states(
        df,
        attention_weights=att_weights,
        feature_cols=FEATURE_COLS,
    )

    return df, df_att


def run_qlearning_variant(
    df: pd.DataFrame,
    state_cols,
    reward_col: str,
    episodes: int = 1000,
    max_steps: int = 50,
    label: str = "",
):
    """
    Helper to run Q-learning on a given DataFrame and reward column.
    Assumes q_learning_env.PricingEnv(df, state_cols, utility_col, max_steps).
    """

    # Create a working copy with a column named "Utility" used as reward
    df_rl = df.copy()
    df_rl["Utility"] = df_rl[reward_col].astype(float)

    env = PricingEnv(
        df=df_rl,
        state_cols=state_cols,
        utility_col="Utility",
        max_steps=max_steps,
    )

    t0 = time.perf_counter()
    q_table, rewards = run_q_learning(
        env,
        n_episodes=episodes,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
    )
    t1 = time.perf_counter()

    rewards = np.array(rewards, dtype=float)
    last_200 = rewards[-200:] if len(rewards) >= 200 else rewards

    result = {
        "Model": label,
        "Episodes": episodes,
        "Mean_Reward_Last200": last_200.mean(),
        "Std_Reward_Last200": last_200.std(ddof=1) if len(last_200) > 1 else 0.0,
        "Total_Training_Time_s": t1 - t0,
    }

    return result, q_table, rewards


def main():
    df, df_att = prepare_data()

    # ---------- Define feature sets ----------
    # Base features (no attention)
    base_state_cols = ["Mileage", "Profit_Margin", "Time_Diff"]

    # Attention-weighted features (from df_att)
    att_state_cols = [f"att_{col}" for col in FEATURE_COLS]

    # ---------- Traditional RL: margin-only reward ----------
    print("\n=== Q-learning: Traditional RL (margin-only) ===")
    res_trad, q_trad, r_trad = run_qlearning_variant(
        df=df,
        state_cols=base_state_cols,
        reward_col="Profit_Margin",
        episodes=1000,
        max_steps=50,
        label="Traditional (Margin-only)",
    )

    # ---------- Utility-based RL (no attention) ----------
    print("\n=== Q-learning: Utility-based RL ===")
    res_util, q_util, r_util = run_qlearning_variant(
        df=df,
        state_cols=base_state_cols,
        reward_col="Utility",
        episodes=1000,
        max_steps=50,
        label="Utility-based RL",
    )

    # ---------- Utility + Attention ----------
    print("\n=== Q-learning: Utility + Attention ===")
    res_att, q_att, r_att = run_qlearning_variant(
        df=df_att,
        state_cols=att_state_cols,
        reward_col="Utility",
        episodes=1000,
        max_steps=50,
        label="Utility + Attention",
    )

    # ---------- Summarize ----------
    results_df = pd.DataFrame([res_trad, res_util, res_att])
    print("\n=== Reward comparison (last 200 episodes) ===")
    print(results_df[["Model", "Mean_Reward_Last200", "Std_Reward_Last200"]])

    here = Path(__file__).resolve().parent
    out_path = here / "reward_comparison_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved reward comparison to: {out_path}")


if __name__ == "__main__":
    main()
