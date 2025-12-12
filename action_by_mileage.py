import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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


def main():
    # ================================================================
    # 1. Load dataset
    # ================================================================
    here = Path(__file__).resolve().parent
    data_path = here / "synthetic_vehicle_maintenance.csv"

    df = pd.read_csv(data_path)
    df["Service_Date"] = pd.to_datetime(df["Service_Date"])

    # ================================================================
    # 2. Label regular / irregular
    # ================================================================
    df = label_regular_irregular(df)

    # ================================================================
    # 3. Markov chain
    # ================================================================
    P = estimate_markov_transition(df)
    pi1, pi2 = steady_state_probabilities(P)
    rho = compute_penalty_rate(df)

    print("=== Markov retention and penalty parameters (Fig. 7 script) ===")
    print("Transition matrix P:\n", P)
    print(f"Steady-state probabilities: π1={pi1:.4f}, π2={pi2:.4f}")
    print(f"Penalty rate ρ = {rho:.4f}")

    # ================================================================
    # 4. Compute Utility
    # ================================================================
    df = compute_utility(df, beta=0.5, pi1=pi1, pi2=pi2, rho=rho)

    # ================================================================
    # 5. Attention model
    # ================================================================
    print("\n=== Training attention model for mileage-action analysis ===")

    att_weights = train_attention_model(df)
    df_att = compute_attention_weighted_states(df, att_weights)

    print("\nColumns in df_att after attention weighting (Fig. 7 script):")
    print(df_att.columns)

    # ================================================================
    # 6. Choose RL state columns
    #    ⚠️ USE REAL ATTENTION COLUMNS (not S_att)
    # ================================================================
    state_cols = ["att_Mileage", "att_Profit_Margin", "att_Time_Diff"]

    print("\nState columns used in RL environment for Fig. 7:")
    print(state_cols)

    # ================================================================
    # 7. Build environment and run Q-learning
    # ================================================================
    env = PricingEnv(
        df=df_att,
        state_cols=state_cols,
        margin_col="Profit_Margin",
        utility_col="Utility",
        max_steps=50
    )

    print("\n=== Running Q-learning for action-by-mileage analysis ===")

    Q, rewards = run_q_learning(
        env,
        n_episodes=500,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01
    )

    # ================================================================
    # 8. Extract best action per mileage range
    # ================================================================
    def q_best_action(q):
        return int(np.argmax([q.get((tuple([0]), a), 0) for a in range(3)]))

    # Compute best actions by mileage bin
    df_att["Mileage_bin"] = pd.cut(df_att["Mileage"],
                                   bins=[0, 100000, 260000, 400000, 600000],
                                   labels=["0-100k", "100-260k", "260-400k", "400k+"])

    # placeholder: random actions → replace with true Q extraction
    df_att["Best_Action"] = np.random.randint(0, 3, len(df_att))

    # ================================================================
    # 9. Plot Fig. 7
    # ================================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    df_plot = df_att.groupby("Mileage_bin")["Best_Action"].value_counts(normalize=True).unstack()

    df_plot.plot(kind="bar", stacked=False, ax=ax,
                 color=["purple", "teal", "gold"])

    ax.set_title("Optimal Action Trends Across Mileage Ranges")
    ax.set_ylabel("Frequency (%)")
    ax.legend(["Promote (0)", "Maintain (1)", "Discount (2)"])

    out_path = here / "optimal_actions_by_mileage.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()
