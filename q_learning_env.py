import numpy as np


class PricingEnv:
    """
    Reinforcement Learning environment for preventive maintenance pricing.

    States: vector built from selected columns (e.g., attention-weighted features).
    Actions: 0 = Promote, 1 = Maintain, 2 = Discount.
    Reward: utility value per transaction (Utility column).
    """

    def __init__(
        self,
        df,
        state_cols,
        margin_col="Profit_Margin",
        utility_col="Utility",
        max_steps=50,
    ):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Must contain state_cols, margin_col, and utility_col.
        state_cols : list[str]
            Names of columns used as the state representation.
        margin_col : str
            Name of the margin column (kept for consistency with the paper).
        utility_col : str
            Name of the column containing per-transaction utility.
        max_steps : int
            Maximum steps per episode.
        """
        self.df = df.reset_index(drop=True)
        self.state_cols = state_cols
        self.margin_col = margin_col
        self.utility_col = utility_col
        self.max_steps = max_steps

        self.current_step = 0
        self.n_samples = len(self.df)

        # Pre-extract states and rewards from dataframe
        self.states = self.df[self.state_cols].values
        self.rewards = self.df[self.utility_col].values

        self.current_state = None

    def reset(self):
        """Start a new episode by sampling a random state."""
        self.current_step = 0
        idx = np.random.randint(0, self.n_samples)
        self.current_state = self.states[idx]
        return self.current_state

    def step(self, action):
        """
        Take an action and transition to the next random state.

        This is an event-based environment:
        - Next state is sampled from the empirical distribution of states.
        - Reward is sampled from the empirical distribution of utilities.
        """
        # Sample reward from empirical distribution
        reward_idx = np.random.randint(0, self.n_samples)
        reward = float(self.rewards[reward_idx])

        # Update step count
        self.current_step += 1

        # Sample next state
        next_idx = np.random.randint(0, self.n_samples)
        next_state = self.states[next_idx]

        # Terminate at max_steps
        done = self.current_step >= self.max_steps
        self.current_state = next_state

        return next_state, reward, done, {}


def _state_to_key(state):
    """Convert a state vector to a hashable key for the Q-table dictionary."""
    return tuple(state.tolist())


def run_q_learning(
    env,
    n_episodes=500,
    alpha=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
):
    """
    Tabular Q-learning on PricingEnv.

    Returns
    -------
    Q : dict
        Mapping (state_key, action) -> Q-value.
    rewards_per_episode : list[float]
        Total reward per episode.
    """
    Q = {}
    rewards_per_episode = []

    def get_q(s_key, a):
        return Q.get((s_key, a), 0.0)

    def set_q(s_key, a, value):
        Q[(s_key, a)] = value

    # Linear epsilon decay
    epsilons = np.linspace(epsilon_start, epsilon_end, n_episodes)

    for ep in range(n_episodes):
        state = env.reset()
        state_key = _state_to_key(state)
        done = False
        total_reward = 0.0
        epsilon = epsilons[ep]

        while not done:
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(3)  # 3 actions
            else:
                q_values = [get_q(state_key, a) for a in range(3)]
                action = int(np.argmax(q_values))

            # Environment step
            next_state, reward, done, _ = env.step(action)
            next_state_key = _state_to_key(next_state)

            total_reward += reward

            # TD target
            if done:
                td_target = reward
            else:
                next_q_values = [get_q(next_state_key, a) for a in range(3)]
                td_target = reward + gamma * max(next_q_values)

            # Q-learning update
            old_q = get_q(state_key, action)
            new_q = old_q + alpha * (td_target - old_q)
            set_q(state_key, action, new_q)

            state_key = next_state_key

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode
