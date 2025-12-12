"""
main_experiment.py

Main entry point for reproducing the utility-based RL with attention
on the synthetic automotive maintenance dataset.

Pipeline:
1. Load CSV dataset.
2. Label visits as regular / irregular.
3. Estimate Markov retention model and penalty rate.
4. Compute utility for each transaction.
5. Train attention model and obtain attention-weighted state features.
6. Run Q-learning in the pricing environment.
7. Print summary statistics and save Q-table.
"""

import pandas as pd
from pathlib import Path

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

from q_learning_env import (
    PricingEnv,
    run_q_learning,
)


def main():
    # ------------------------------------------------------------------
    # 1. Load existing dataset
    # ------------------------------------------------------------------
    here = Path(__file__).resolve().parent
    data_path = here / "synthetic_vehicle_maintenance.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    # parse dates for Markov estimation
    if "Service_Date" in df.columns:
        df["Service_Date"] = pd.to_datetime(df["Service_Date"])

    # ------------------------------------------------------------------
    # 2. Label visits as regular / irregular
    # ------------------------------------------------------------------
    df = label_regular_irregular(
        df,
        max_mileage_diff=10_000,
        max_time_years=1.0,
    )
    # adds "Regular_Flag"

    # ------------------------------------------------------------------
    # 3. Markov retention model & steady-state probabilities
    # ------------------------------------------------------------------
    P = estimate_markov_transition(df)
    pi1, pi2 = steady_state_probabilities(P)

    # ------------------------------------------------------------------
    # 4. Penalty rate ρ from margin statistics and CLV approximation
    # ------------------------------------------------------------------
    rho = compute_penalty_rate(
        df,
        r=0.1,
        T=8,
    )

    print("=== Markov retention and penalty parameters ===")
    print(f"Transition matrix P:\n{P}")
    print(f"Steady-state probabilities: π1={pi1:.4f}, π2={pi2:.4f}")
    print(f"Penalty rate ρ = {rho:.4f}")

    # ------------------------------------------------------------------
    # 5. Compute utility for each transaction
    # ------------------------------------------------------------------
    beta = 0.5

    df = compute_utility(
        df,
        beta=beta,
        pi1=pi1,
        pi2=pi2,
        rho=rho,
    )
    # adds "Immediate_U", "Future_U", "Utility"

    # ------------------------------------------------------------------
    # 6. Train attention model & compute attention-weighted states
    # ------------------------------------------------------------------
    print("\n=== Training attention model ===")
    att_weights = train_attention_model(df)

    df_att = compute_attention_weighted_states(
        df,
        attention_weights=att_weights,
        feature_cols=FEATURE_COLS,
        out_prefix="att_",
    )

    print("\nColumns in df_att after attention weighting:")
    print(df_att.columns)

    # ------------------------------------------------------------------
    # 7. Build RL environment and run Q-learning
    # ------------------------------------------------------------------
    state_cols = [f"att_{col}" for col in FEATURE_COLS]
    print("\nState columns used in RL environment:", state_cols)

    env = PricingEnv(
        df=df_att,
        state_cols=state_cols,
        utility_col="Utility",
        max_steps=50,
    )

    print("\n=== Running Q-learning ===")
    q_table, rewards = run_q_learning(
        env,
        n_episodes=500,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
    )

    # ------------------------------------------------------------------
    # 8. Summary statistics & save Q-table
    # ------------------------------------------------------------------
    rewards_series = pd.Series(rewards)
    print("\n=== Q-learning training summary ===")
    print(f"Number of episodes: {len(rewards)}")
    print(f"Mean reward (all episodes): {rewards_series.mean():.2f}")
    print(f"Mean reward (last 100 episodes): {rewards_series.tail(100).mean():.2f}")
    print(f"Std reward (last 100 episodes): {rewards_series.tail(100).std():.2f}")

    # Convert Q dict to DataFrame
    q_records = []
    for (state_key, action), value in q_table.items():
        q_records.append(
            {
                "state": state_key,
                "action": action,
                "q_value": value,
            }
        )
    q_df = pd.DataFrame(q_records)

    q_path = here / "q_table.csv"
    q_df.to_csv(q_path, index=False)
    print(f"\nQ-table saved to: {q_path}")


if __name__ == "__main__":
    main()
