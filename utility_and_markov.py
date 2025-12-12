# src/utility_and_markov.py

import numpy as np
import pandas as pd

def label_regular_irregular(df: pd.DataFrame,
                            max_mileage_diff: int = 10_000,
                            max_time_years: float = 1.0) -> pd.DataFrame:
    """
    Add Regular_Flag column:
    1 = regular (Mileage_Diff <= 10k OR Time_Diff <= 1 year)
    0 = irregular otherwise.
    """
    regular = (
        (df["Mileage_Diff"] <= max_mileage_diff) |
        (df["Time_Diff"] <= max_time_years)
    )
    df = df.copy()
    df["Regular_Flag"] = regular.astype(int)
    return df


def estimate_markov_transition(df: pd.DataFrame) -> np.ndarray:
    """
    Estimate 2x2 Markov transition matrix P_ij from Regular_Flag
    over consecutive visits per customer. States: 0 = irregular (S2), 1 = regular (S1).
    """
    df_sorted = df.sort_values(["Customer_ID", "Service_Date"])
    flags = df_sorted["Regular_Flag"].values
    cust = df_sorted["Customer_ID"].values

    counts = np.zeros((2, 2), dtype=float)  # i->j

    for i in range(len(df_sorted) - 1):
        if cust[i] != cust[i + 1]:
            continue  # new episode
        s_i = flags[i]
        s_j = flags[i + 1]
        counts[s_i, s_j] += 1

    # avoid division by zero
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums
    return P


def steady_state_probabilities(P: np.ndarray):
    """
    Compute steady-state distribution (π0, π1) for 2-state chain from Eq. (11)-(12).
    State 1 = regular (S1), state 0 = irregular (S2).
    """
    P11 = P[1, 1]
    P12 = P[1, 0]
    P21 = P[0, 1]
    P22 = P[0, 0]

    # π1 = P21 / (P21 + P12), π2 = P12 / (P21 + P12)
    denom = P21 + P12
    if denom == 0:
        # fallback: 50/50 if degenerate
        return 0.5, 0.5
    pi1 = P21 / denom  # regular
    pi2 = P12 / denom  # irregular
    return pi1, pi2


def compute_penalty_rate(df: pd.DataFrame,
                         r: float = 0.1,
                         T: int = 8) -> float:
    """
    Compute penalty rate ρ = (Clost + Cret) / M_reg as in Eq. (5).
    CLV approximated using average margins and Markov chain retention. :contentReference[oaicite:10]{index=10}
    """
    reg = df[df["Regular_Flag"] == 1]["Profit_Margin"].values
    irreg = df[df["Regular_Flag"] == 0]["Profit_Margin"].values

    M_reg = reg.mean() if len(reg) > 0 else df["Profit_Margin"].mean()
    M_irreg = irreg.mean() if len(irreg) > 0 else 0.8 * M_reg  # fallback

    C_lost = M_reg - M_irreg

    # Simple approximate CLV difference: geometric series with different retention probs
    # CLV_reg = sum_{t=1..T} p_reg^t * M_reg / (1+r)^t, same for irregular.
    # For simplicity, assume p_reg > p_irreg and estimate from data frequency ratio.
    n_reg_cust = df[df["Regular_Flag"] == 1]["Customer_ID"].nunique()
    n_irreg_cust = df[df["Regular_Flag"] == 0]["Customer_ID"].nunique()
    total_cust = df["Customer_ID"].nunique()

    p_reg = n_reg_cust / total_cust if total_cust > 0 else 0.7
    p_irreg = n_irreg_cust / total_cust if total_cust > 0 else 0.3

    disc_factors = np.array([(p_reg ** t) / ((1 + r) ** t) for t in range(1, T + 1)])
    CLV_reg = (disc_factors * M_reg).sum()

    disc_factors_ir = np.array([(p_irreg ** t) / ((1 + r) ** t) for t in range(1, T + 1)])
    CLV_irreg = (disc_factors_ir * M_irreg).sum()

    C_ret = CLV_reg - CLV_irreg

    rho = (C_lost + C_ret) / M_reg if M_reg > 0 else 0.0
    rho = max(0.0, min(rho, 0.95))  # keep in [0,1)
    return rho


def compute_utility(df: pd.DataFrame,
                    beta: float,
                    pi1: float,
                    pi2: float,
                    rho: float) -> pd.DataFrame:
    """
    Add columns Immediate, Future, Utility as in Eqs. (16)-(18). :contentReference[oaicite:11]{index=11}
    Regular_Flag = 1 => S1, Regular_Flag = 0 => S2.
    """
    df = df.copy()
    M = df["Profit_Margin"].values
    is_reg = df["Regular_Flag"].values == 1

    # Immediate I(S_t)
    I = np.where(is_reg, M, M * (1 - rho))

    # Future F(S_t)
    F = np.where(is_reg, pi1 * M, pi2 * M * (1 - rho))

    U = I + beta * F

    df["Immediate_U"] = I
    df["Future_U"] = F
    df["Utility"] = U
    return df
