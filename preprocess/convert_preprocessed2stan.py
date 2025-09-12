import pandas as pd
import numpy as np
from pathlib import Path


def read_preprocessed_data():
    """Read preprocessed data from CSV."""
    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Preprocessed data file not found: {PREPROCESSED_DATA_PATH}")
    return pd.read_csv(PREPROCESSED_DATA_PATH)


def _choice_to_one_based(choice):
    """Convert zero-based choice to one-based choice. If choice is None, change it to -1"""
    is_nan = np.isnan(choice)
    choice = choice + 1
    choice[is_nan] = -1  # Convert NaN to -1
    return choice

def _block_to_zero_based(block):
    """Convert block number to zero-based index. If block is None, change it to -1"""
    block = block - 1
    return block

def convert_to_stan_format(df):
    """Convert preprocessed data to a format suitable for Stan.

    Expects columns:
      id, block, trial, partner_choice, partner_reward, self_choice, is_partner_high_exp

    Returns:
      dict with keys:
        n_subjects, n_blocks, n_trials, n_choices,
        self_choices[S,B,T] (int),
        other_choices[S,B,T] (int),
        other_rewards[S,B,T] (int),
        is_partner_high_exp[S,B] (int; per subject-block),
      plus mapping dicts (useful for debugging).
    """

    required_columns = [
        'id', 'block', 'trial',
        'partner_choice', 'partner_reward',
        'self_choice', 'is_partner_high_exp'
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Coerce choice columns to float first so np.isnan works; then to 1-based
    df['partner_choice'] = _choice_to_one_based(df['partner_choice'].astype(float))
    df['self_choice']    = _choice_to_one_based(df['self_choice'].astype(float))

    # Coerce block to zero-based index
    df['block'] = _block_to_zero_based(df['block'].astype(float))

    n_subjects = df['id'].nunique()
    n_blocks = df['block'].nunique()
    n_trials = df['trial'].nunique()
    n_choices = 3
    
    subj_levels  = sorted(df['id'].unique().tolist())
    subj_to_idx  = {s: i for i, s in enumerate(subj_levels,  start=0)}

    # Allocate arrays: fill with -1 to mark missing; rewards default to -1 too
    self_choices  = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)
    other_choices = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)
    other_rewards = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)
    is_high_exp   = np.full((n_subjects, n_blocks),           -1, dtype=int)

    # Populate per (id, block, trial)
    for _, row in df.iterrows():
        si = subj_to_idx[int(row["id"])]
        bi = int(row["block"])
        ti = int(row['trial'])

        # Choices are already 1-based with -1 for missing
        other_choices[si, bi, ti] = int(row['partner_choice'])
        self_choices[si, bi, ti]  = int(row['self_choice'])

        # Rewards: keep as int; if NaN, set -1
        rew = row['partner_reward']
        if pd.isna(rew):
            other_rewards[si, bi, ti] = -1
        else:
            other_rewards[si, bi, ti] = int(round(float(rew)))

        # Per-(subject, block) flag; enforce single value per block
        is_high_exp[si, bi] = int(row['is_partner_high_exp']) + 1
    stan_data = {
        "n_subjects": n_subjects,
        "n_blocks": n_blocks,
        "n_trials": n_trials,
        "n_choices": n_choices,
        "self_choices": self_choices.tolist(),
        "other_choices": other_choices.tolist(),
        "other_rewards": other_rewards.tolist(),
        "noise_level_condition": is_high_exp.tolist(),  # include if your Stan model uses it
        "idx2subject": {i: s for s, i in subj_to_idx.items()},
    }

    return stan_data

# Example “main” usage:
if __name__ == "__main__":
    debug = False
    PREPROCESSED_DATA_PATH = Path("../data/preprocessed/preprocessed_data.csv").resolve()
    OUTPUT_PATH = Path("../data/stan_model/stan_data.json").resolve()
    if debug:
        PREPROCESSED_DATA_PATH = Path("test_data.csv")
        OUTPUT_PATH = Path("test_stan_data.json")

    print(f"Reading preprocessed data from {PREPROCESSED_DATA_PATH}")
    print(f"Will write Stan data to {OUTPUT_PATH}")
    df = read_preprocessed_data()
    stan_data = convert_to_stan_format(df)
    with open(OUTPUT_PATH, 'w') as f:
        import json
        json.dump(stan_data, f, indent=2)
        print(f"Saved Stan data to {OUTPUT_PATH}")