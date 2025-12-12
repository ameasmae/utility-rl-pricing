import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_fig(
    utility_csv="mean_abs_shap_utility_only.csv",
    att_csv="mean_abs_shap_utility_attention.csv",
    out_png="fig_shap_comparison.png",
):
    # Load mean(|SHAP|) summaries produced by shap_instability.py
    base = pd.read_csv(utility_csv, index_col=0)
    att  = pd.read_csv(att_csv, index_col=0)

    # Convert to Series
    base_s = base["mean_abs_shap"]
    att_s  = att["mean_abs_shap"]

    # Align common features for clean comparison (ignore S_att in left plot if you want)
    common = [c for c in ["Mileage", "Profit_Margin", "Time_Diff"] if c in base_s.index and c in att_s.index]

    # Left plot: two bars per feature (utility-only vs utility+att)
    x = np.arange(len(common))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].bar(x - w/2, base_s.loc[common].values, width=w, label="Utility-only")
    axes[0].bar(x + w/2, att_s.loc[common].values, width=w, label="Utility + Attention")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(common, rotation=0)
    axes[0].set_title("Mean Absolute SHAP Values (Comparison)")
    axes[0].set_ylabel("Mean |SHAP|")
    axes[0].legend()

    # Right plot: difference (attention - baseline)
    diff = att_s.loc[common] - base_s.loc[common]
    axes[1].bar(x, diff.values)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(common, rotation=0)
    axes[1].axhline(0, linewidth=1)
    axes[1].set_title("Δ Mean |SHAP| (Attention − Baseline)")
    axes[1].set_ylabel("Difference")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved Fig. 5-like plot to: {out_png}")

if __name__ == "__main__":
    plot_fig5()
