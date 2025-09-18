import numpy as np
import pandas as pd
from pathlib import Path


class Setting:
    BASE_DIR = Path(__file__).resolve().parent # Directory of the current script
    BEHAVIORAL_DATA_PATH = (BASE_DIR / "../../../data/preprocessed/preprocessed_data.csv").resolve()

    OUT_DIR_PATH = BASE_DIR / "results/"

    @staticmethod
    def create_directories():
        Setting.OUT_DIR_PATH.mkdir(exist_ok=True)


def read_data():
    return pd.read_csv(Setting.BEHAVIORAL_DATA_PATH)


def compute_ave_correct_rate(data):
    data["is_correct"] = data["self_choice"] == 2  # 2 is the best, 0 is the worst
    grouped = data.groupby(["id", "is_partner_high_exp"])
    choice_data_w_high_exp = []
    choice_data_w_low_exp = []
    for (p_id, cond), choice in grouped:
        average_correct = choice.groupby("trial").mean()["is_correct"]
        if cond:
            choice_data_w_high_exp.append(average_correct)
        else:
            choice_data_w_low_exp.append(average_correct)

    choice_data_w_high_exp = np.array(choice_data_w_high_exp)
    choice_data_w_low_exp = np.array(choice_data_w_low_exp)

    mean_correct_high = choice_data_w_low_exp.mean(axis=0)
    se_correct_high = np.std(choice_data_w_low_exp, axis=0)

    mean_correct_low = choice_data_w_low_exp.mean(axis=0)
    se_correct_low = np.std(choice_data_w_low_exp, axis=0)

    return (mean_correct_high, se_correct_high / 37, mean_correct_low, se_correct_low  / 37)


def main():
    choice_data = read_data()
    mean_correct_high, se_correct_high, mean_correct_low, se_correct_low = compute_ave_correct_rate(choice_data)
    pd.DataFrame({"mean": mean_correct_high, "se": se_correct_high}).to_csv(Setting.OUT_DIR_PATH / "trial_correct_rates_w_high.csv", index=False)
    pd.DataFrame({"mean": mean_correct_low, "se": se_correct_low}).to_csv(Setting.OUT_DIR_PATH / "trial_correct_rates_w_low.csv", index=False)


if __name__ == "__main__":
    Setting.create_directories()
    main()