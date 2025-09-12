import pandas as pd
from pathlib import Path
from glob import glob
import numpy as np

class Setting:
    BASE_DIR = Path(__file__).resolve().parent # Directory of the current script
    MODEL_PARAM_PATH = BASE_DIR / "../../computational_analysis/fit_const_weight_full_real_data/results/fit_summary.csv"
    BEHAVIORAL_DATA_DIR = BASE_DIR / "../../../data/behavioral_data/"
    EVENT_DATA_PATH = BASE_DIR / "../../../data/events/"
    SUBJECTS = list(range(1, 19)) + list(range(20, 38))  # Exclude subject 19 due to missing data


def extract_learning_rates(subject_id, partner_noise):
    """
    subject_id: int, the ID of the subject (1-30)
    partner_noise: int, 1 for low noise partner, 2 for high noise partner
    """
    # Read the model fitting results
    df = pd.read_csv(Setting.MODEL_PARAM_PATH, index_col=0)

    # Extract the learning rate for the given subject
    if subject_id >= 19:
        subject_id -= 1
    lr_reward = df.loc[f"lr_reward[{subject_id},{partner_noise}]"]["Mean"]
    lr_action = df.loc[f"lr_action[{subject_id},{partner_noise}]"]["Mean"]
    return lr_reward, lr_action


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
    # pes = np.array(pes)
    # pes = (pes - np.mean(pes)) / np.std(pes)

    events = pd.DataFrame({"onset": onsets, "duration": 0.0, "trial_type": "rpe", "modulation": pes})
    return events


def create_events_rp_init_q0(choice_data, lr):
    values = np.zeros(3)
    partner_choice = choice_data["partner_choice"].tolist()
    partner_reward = choice_data["partner_reward"].tolist()
    onsets = choice_data["t_other_outcome"].tolist()
    pes = []
    for c, r in zip(partner_choice, partner_reward):
        pe = r - values[c]
        values[c] += lr * pe
        pes.append(pe)

    events = pd.DataFrame({"onset": onsets, "duration": 0.0, "trial_type": "rpe_q0", "modulation": pes})
    return events


# Not used anymore
def create_events_value_reward(choice_data, lr):
    values = np.ones(3) / 2
    partner_choice = choice_data["partner_choice"].tolist()
    partner_reward = choice_data["partner_reward"].tolist()
    onsets = choice_data["t_other_outcome"].tolist()
    vals = []
    for c, r in zip(partner_choice, partner_reward):
        val = values[c]
        vals.append(val)
        values[c] += lr * (r - values[c])

    # Z-standarize the values
    # vals = np.array(vals)
    # vals = (vals - np.mean(vals)) / np.std(vals)

    # Z-standarize the rewards
    partner_reward = np.array(partner_reward)
    partner_reward = (partner_reward - np.mean(partner_reward)) / np.std(partner_reward)

    events = pd.DataFrame({"onset": onsets, "duration": 0.0, "trial_type": "reward", "modulation": partner_reward})
    events2 = pd.DataFrame({"onset": onsets, "duration": 0.0, "trial_type": "value", "modulation": vals})
    return pd.concat([events, events2], ignore_index=True)


def create_events_ape(choice_data, lr):
    action_tendencies = np.ones(3) / 3
    partner_choice = choice_data["partner_choice"].tolist()
    onsets = choice_data["t_other_choice"].tolist()

    pes = []
    for c in partner_choice:
        pe = 1 - action_tendencies[c]
        action_tendencies[c] += lr * pe
        for i in range(3):
            if i != c:
                action_tendencies[i] += lr * (0 - action_tendencies[i])
        pes.append(pe)
    
    # Z-standarize the prediction errors
    # pes = np.array(pes)
    # pes = (pes - np.mean(pes)) / np.std(pes)

    events = pd.DataFrame({"onset": onsets, "duration": 0.0, "trial_type": "ape", "modulation": pes})
    return events

def create_events_others_outcome(choice_data):
    onsens = choice_data["t_other_outcome"].tolist()
    partner_rewards = choice_data["partner_reward"]
    rewarded = partner_rewards == 1
    rewarded_onsets = onsens[rewarded].tolist()
    rewarded_events = pd.DataFrame({"onset": rewarded_onsets, "duration": 0.0, "trial_type": "other_rewarded_outcome", "modulation": 1})
    unrewarded = partner_rewards == 0
    unrewarded_onsets = onsens[unrewarded].tolist()
    unrewarded_events = pd.DataFrame({"onset": unrewarded_onsets, "duration": 0.0, "trial_type": "other_non_rewarded_outcome", "modulation": 1})
    events = pd.concat([rewarded_events, unrewarded_events], ignore_index=True)
    return events


def create_events_self_options(choice_data):
    onsets_self_choice = choice_data["t_self_choice_on"]
    onsets_self_options = choice_data["t_self_options_on"]
    rt = onsets_self_choice - onsets_self_options

    events = pd.DataFrame({"onset": onsets_self_options, "duration": rt.tolist(), "trial_type": "self_options", "modulation": 1})
    return events


def create_events(choice_data, name, duration):
    onsens = choice_data[f"t_{name}"].tolist()
    events = pd.DataFrame({"onset": onsens, "duration": duration, "trial_type": name, "modulation": 1})
    return events


def main():
    conditions = {"subject_id": [], "run": [], "condition": []}
    reward_prediction_errors = {"subject_id": [], "run": [], "trial": [], "rpe": []}
    action_prediction_errors = {"subject_id": [], "run": [], "trial": [], "ape": []}

    for subject_id in Setting.SUBJECTS:
        if subject_id == 19: # subject 19 has no MRI data
            continue
        for run in range(1, 5):
            condition = extract_condition(subject_id, run)
            conditions["subject_id"].append(subject_id)
            conditions["run"].append(run)
            conditions["condition"].append(condition)

            lr_reward, lr_action = extract_learning_rates(subject_id, condition)
            choice_data = extract_choice_data(subject_id, run)
            
            rpe = create_events_rpe(choice_data, lr_reward)
            rpe_q0 = create_events_rp_init_q0(choice_data, lr_reward)
            ape = create_events_ape(choice_data, lr_action)

            reward_prediction_errors["subject_id"].extend([subject_id] * len(rpe))
            reward_prediction_errors["run"].extend([run] * len(rpe))
            reward_prediction_errors["trial"].extend(list(range(1, len(rpe) + 1)))
            reward_prediction_errors["rpe"].extend(rpe["modulation"].tolist())

            action_prediction_errors["subject_id"].extend([subject_id] * len(ape))
            action_prediction_errors["run"].extend([run] * len(ape))
            action_prediction_errors["trial"].extend(list(range(1, len(ape) + 1)))
            action_prediction_errors["ape"].extend(ape["modulation"].tolist())
            
            set_events = [rpe]
            set_events.append(ape)
            set_events.append(rpe_q0)
            # set_events.append(create_events_value_reward(choice_data, lr_reward))
            set_events.append(create_events_self_options(choice_data))
            for name, duration in [("other_options", 2.0), ("other_choice", 0.0), ('other_outcome', 0.0), ("self_choice_on", 1.0)]:
                set_events.append(create_events(choice_data, name, duration))

            events = pd.concat(set_events, ignore_index=True)
            events.dropna(subset=["onset"], inplace=True)
            events.sort_values(by="onset", inplace=True)
                
            save_path = Setting.EVENT_DATA_PATH / f"sub-{subject_id:03d}" 
            save_path.mkdir(parents=True, exist_ok=True)
            events.to_csv(save_path / f"sub-{subject_id:03d}_task-ol_run-{run:02d}_events.tsv", sep="\t", index=False)
            print(f"Saved events for subject {subject_id}, run {run}")

    preprocessed_dir = (Setting.BASE_DIR / "../../../data/preprocessed").resolve()

    pd.DataFrame(conditions).to_csv(preprocessed_dir / "conditions.csv", index=False)
    pd.DataFrame(reward_prediction_errors).to_csv(preprocessed_dir / "reward_prediction_errors.csv", index=False)
    pd.DataFrame(action_prediction_errors).to_csv(preprocessed_dir / "action_prediction_errors.csv", index=False)


if __name__ == "__main__":
    main()
