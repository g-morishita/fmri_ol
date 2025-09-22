import pandas as pd
import numpy as np
from pathlib import Path


class Setting:
    BASE_PATH = Path(__file__).parent.resolve()
    FIGURE_PATH = BASE_PATH / "figures"
    FIGURE_PATH.mkdir(exist_ok=True, parents=True)
    RESULTS_PATH = BASE_PATH / "results"


def read_data(noise_condition):
    file_path = Setting.RESULTS_PATH / f'simulation_lr_0.3_beta_{"high" if noise_condition == "high" else "low"}.csv'
    return pd.read_csv(file_path)


def compute_learning_rate(data):
    data = data.copy()
    data['is_correct'] = (data["choice"] == 2)

    learning_rate = data.groupby(["trial"]).mean()["is_correct"].reset_index()

    return learning_rate

def plot_learning_rate(learning_rate_high, learning_rate_low):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(learning_rate_high['trial'], learning_rate_high['is_correct'], label='High Noise', color='orange')
    plt.plot(learning_rate_low['trial'], learning_rate_low['is_correct'], label='Low Noise', color='green')
    plt.xticks([0, 14, 29], labels=["1", "15", "30"], fontsize=18)
    plt.yticks([0, 0.5, 1], labels=["0", "0.5", "1"], fontsize=18)
    plt.hlines(0.3, 0, 29, colors='gray', linestyles='dashed', label='chance level')
    plt.title("")
    plt.xlim(0, 29)
    plt.ylim(0, 1)
    plt.legend(prop={'size': 18})
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(Setting.FIGURE_PATH / 'learning_rate_comparison.png')
    plt.show()
    plt.close()

def main():
    data_high = read_data('high')
    data_low = read_data('low')

    learning_rate_high = compute_learning_rate(data_high)
    learning_rate_low = compute_learning_rate(data_low)

    plot_learning_rate(learning_rate_high, learning_rate_low)


if __name__ == "__main__":
    main()
