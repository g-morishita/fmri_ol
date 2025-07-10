import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from analysis.computational_analysis.nll import nll, nll_per_session_full
from pathlib import Path

DATA_PATH = Path("preprocess/preprocessed_data.csv")

def read_data():
    df = pd.read_csv(str(DATA_PATH))
    return df


def create_subject_data_for_each_condition(df):
    subject_df_high_exp = df.groupby(["id", "block", "is_partner_high_exp"])
    data = {}
    for (id, block, is_partner_high_exp), block_df in subject_df_high_exp:
        if id not in data:
            data[id] = {}
        if is_partner_high_exp not in data[id]:
            # Initialize the dictionary structure for this condition
            data[id][True] = {"partner_reward": [], "partner_choice": [], "self_choice": []}
            data[id][False] = {"partner_reward": [], "partner_choice": [], "self_choice": []}

        # Extract values from the DataFrame
        partner_reward = block_df["partner_reward"].values
        partner_choice = block_df["partner_choice"].values
        self_choices = block_df["self_choice"].values

        # Append values to the respective lists
        data[id][is_partner_high_exp]["partner_reward"].append(partner_reward.tolist())
        data[id][is_partner_high_exp]["partner_choice"].append(partner_choice.tolist())
        data[id][is_partner_high_exp]["self_choice"].append(self_choices.tolist())
    
    return data


def fit_full(set_other_rewards, set_other_choices, set_self_choices):
    obj_func = lambda params: nll(params, nll_per_session_full, set_other_rewards, set_other_choices, set_self_choices)

    min_nll = np.inf
    best_x = np.nan
    for _ in range(30):
        x0 = [np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform(0, 100)]
        bounds = Bounds([0, 0, 0, 0], [1, 1, 1, 100])
        res = minimize(obj_func, x0, options={"maxiter": 1000, "disp": False}, bounds=bounds)
        if res.success:
            if min_nll > res.fun:
                min_nll = res.fun
                best_x = res["x"]

    return best_x, min_nll


def main():
    df = read_data()
    data = create_subject_data_for_each_condition(df)
    results = []
    for id, blocks in data.items():
        for is_partner_high_exp, block_data in blocks.items():
            set_other_rewards = block_data["partner_reward"]
            set_other_choices = block_data["partner_choice"]
            set_self_choices = block_data["self_choice"]
            best_params, min_nll = fit_full(set_other_rewards, set_other_choices, set_self_choices)
            results.append({
                "id": id,
                "is_partner_high_exp": is_partner_high_exp,
                "best_params": best_params,
                "min_nll": min_nll
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results_full_model.csv", index=False)

if __name__ == "__main__":
    main()