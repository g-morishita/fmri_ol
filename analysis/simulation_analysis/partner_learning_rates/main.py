import pandas as pd
import numpy as np
from pathlib import Path


class Setting:
    BASE_PATH = Path(__file__).parent.resolve()
    RESULTS_PATH = BASE_PATH / "results"
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    N_TRIALS = 30
    N_SIM = 1_000
    REWARD_PROBS = [0.25, 0.5, 0.75]
    LR = 0.3
    HIGH_NOISE_BETA = 1.5
    LOW_NOISE_BETA = 20.0


def simulate_choice_data(lr, beta, n_trials):
    values = np.zeros(3)
    choices = []
    rewards = []

    for _ in range(n_trials):
        exp_values = np.exp(beta * values)
        probs = exp_values / np.sum(exp_values)
        choice = np.random.choice(3, p=probs)
        reward = int(Setting.REWARD_PROBS[choice] > np.random.rand())
        choices.append(choice)
        rewards.append(reward)
    
        values[choice] += lr * (reward - values[choice])
    
    return np.array(choices), np.array(rewards)


def run_simulation(lr, beta, n_trials, n_sim):
    all_choices = []
    all_rewards = []

    for _ in range(n_sim):
        choices, rewards = simulate_choice_data(lr, beta, n_trials)
        all_choices.append(choices)
        all_rewards.append(rewards)

    return np.array(all_choices), np.array(all_rewards)


def main():
    for beta in [Setting.HIGH_NOISE_BETA, Setting.LOW_NOISE_BETA]:
        choices, rewards = run_simulation(Setting.LR, beta, Setting.N_TRIALS, Setting.N_SIM)
        df = pd.DataFrame({
            'simulation': np.repeat(np.arange(Setting.N_SIM), Setting.N_TRIALS),
            'trial': np.tile(np.arange(Setting.N_TRIALS), Setting.N_SIM),
            'choice': choices.flatten(),
            'reward': rewards.flatten()
        })
        beta_label = "high" if beta == Setting.HIGH_NOISE_BETA else "low"
        df.to_csv(Setting.RESULTS_PATH / f'simulation_lr_{Setting.LR}_beta_{beta_label}.csv', index=False)


if __name__ == "__main__":
    main()