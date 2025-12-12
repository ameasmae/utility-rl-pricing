# utility-rl-pricing
# Utility-Based Reinforcement Learning with Attention for Pricing Optimization

This repository provides a reproducible implementation of the framework proposed in:

**Utility-Based Reinforcement Learning with Attention Mechanisms for Pricing Optimization in Automotive Maintenance Services**

---

## Overview

The framework combines:
- Utility-based reward modeling (profitability + retention)
- Tabular Q-learning
- Attention-based feature weighting
- SHAP explainability and stability analysis

Due to confidentiality constraints, a paper-faithful **synthetic dataset** is provided.

---

## Repository Structure

- synthetic_vehicle_maintenance.csv  # Synthetic dataset
- utility_and_markov.py              # Utility & Markov modeling
- attention_model.py                 # Attention mechanism
- q_learning_env.py                  # RL environment
- main_experiment.py                 # Main pipeline
- reward_comparison.py               # Reward evaluation
- shap_analysis.py                   # SHAP explainability
- shap_instability.py                # SHAP stability analysis
- action_by_mileage.py               # Analysis of optimal pricing actions across mileage ranges



