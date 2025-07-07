import numpy as np
import pandas as pd
from pathlib import Path


DATA_ROOT = Path("../data/behavioral_data")


def extract_participant_ids():
    ids = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
    ids.sort()
    return ids


def extract_block_conditions(id):
    condition_path = DATA_ROOT / id / "condition/block_for_male.csv"
    condition = pd.read_csv(str(condition_path)).iloc[1:]  # Remove the practice session (1st block)
    condition["is_partner_high_exp"] = condition["trial_path"].str.contains("high")
    condition = condition[["block", "is_partner_high_exp"]]

    return condition


def extract_choice_data(id):
    dfs = []
    data_path = DATA_ROOT / id
    for i in range(1, 5):
        for file in data_path.glob(f"*_block={i}.csv"):
            df = pd.read_csv(file).assign(id=id, block=i)
            df["rt"] = df["t_self_options_on"] - df["t_self_highlight_on"]  # Response time = time options are shown - time a chosen option is highlighted
            df["self_choice"] = df["self_choice_stim_idx"]
            df = df[["id", "block", "partner_choice", "partner_reward", "self_choice", "rt"]]
            dfs.append(df)

    choice_data = pd.concat(dfs, ignore_index=True)

    return choice_data


def combine_all_data():
    dfs = []
    ids = extract_participant_ids()
    for id in ids:
        condition = extract_block_conditions(id)
        choice_data = extract_choice_data(id)

        df = choice_data.merge(condition, on="block", how="left")
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main():
    data = combine_all_data()
    data.to_csv("preprocessed_data.csv", index=False)


if __name__ == "__main__":
    main()