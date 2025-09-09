import pandas as pd
from pathlib import Path
from glob import glob
import numpy as np

class Setting:
    BASE_DIR = Path(__file__).resolve().parent # Directory of the current script
    MODEL_PARAM_PATH = BASE_DIR / "../../computational_analysis/fit_const_weight_full_real_data/results/fit_summary.csv"
    BEHAVIORAL_DATA_DIR = BASE_DIR / "../../../data/behavioral_data/"
    EVENT_DATA_PATH = BASE_DIR / "../../../data/events/"
    SUBJECTS = list(range(1, 19)) + list(range(20, 31))  # Exclude subject 19 due to missing data


def extract_learning_rate(subject_id, partner_noise):
    """
    subject_id: int, the ID of the subject (1-30)
    partner_noise: int, 1 for low noise partner, 2 for high noise partner
    """
    # Read the model fitting results
    df = pd.read_csv(Setting.MODEL_PARAM_PATH, index_col=0)

    # Extract the learning rate for the given subject
    if subject_id >= 19:
        subject_id -= 1
    lr = df.loc[f"lr_reward[{subject_id},{partner_noise}]"]["Mean"]
    return lr


def extract_condition(subject_id, run):
    """
    subject_id: int, the ID of the subject (1-30)
    run: int, 1, 2, 3, or 4

    Returns:
    condition: int, 1 for low noise partner, 2 for high noise partner
    """
    condition_df = pd.read_csv(Setting.BEHAVIORAL_DATA_DIR / f"{subject_id:04d}" / "condition" / "block_for_male.csv", index_col=0)
    condition = int("high" in condition_df.loc[run, "trial_path"]) + 1
    return condition


def extract_choice_data(subject_id, run):
    data_path = Setting.BEHAVIORAL_DATA_DIR / f"{subject_id:04d}" / f"*block={run}.csv"
    file = glob(str(data_path))
    if len(file) > 1:
        raise ValueError(f"Multiple files found for subject {subject_id}, run {run}")
    df = pd.read_csv(file[0])
    return df


def create_events_rpe(choice_data, lr):
    values = np.ones(3) / 2
    partner_choice = choice_data["partner_choice"].tolist()
    partner_reward = choice_data["partner_reward"].tolist()
    onsets = choice_data["t_other_outcome"].tolist()
    pes = []
    for c, r in zip(partner_choice, partner_reward):
        pe = r - values[c]
        values[c] += lr * pe
        pes.append(pe)

    # Z-standarize the prediction errors
    pes = np.array(pes)
    pes = (pes - np.mean(pes)) / np.std(pes)

    events = pd.DataFrame({"onset": onsets, "duration": 0.0, "trial_type": "rpe", "modulation": pes})
    return events


def create_events(choice_data, name):
    onsens = choice_data[f"t_{name}"].tolist()
    events = pd.DataFrame({"onset": onsens, "duration": 0.0, "trial_type": name})
    return events


def main():
    for subject_id in Setting.SUBJECTS:
        if subject_id == 19: # subject 19 has no MRI data
            continue
        for run in range(1, 5):
            condition = extract_condition(subject_id, run)
            lr = extract_learning_rate(subject_id, condition)
            choice_data = extract_choice_data(subject_id, run)
            
            set_events = [create_events_rpe(choice_data, lr)]
            for name in ["self_choice_on", "other_choice"]:
                set_events.append(create_events(choice_data, name))

            events = pd.concat(set_events, ignore_index=True)
            events.sort_values(by="onset", inplace=True)
                
            save_path = Setting.EVENT_DATA_PATH / f"sub-{subject_id:03d}" 
            save_path.mkdir(parents=True, exist_ok=True)
            events.to_csv(save_path / f"sub-{subject_id:03d}_task-ol_run-{run:02d}_events.tsv", sep="\t", index=False)
            print(f"Saved events for subject {subject_id}, run {run}")


if __name__ == "__main__":
    main()
