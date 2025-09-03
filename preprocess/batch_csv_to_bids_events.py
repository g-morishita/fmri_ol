#!/usr/bin/env python3
"""
Batch convert behavioral CSVs (organized by subject folders) into BIDS-valid events.tsv.

Assumes folder structure like:
~/PycharmProjects/ol_fmri/data/behavioral_data/
    0001/
        20250702-153635_block=0.csv
        20250702-153635_block=1.csv
        ...
    0002/
        ...
    Demographics.xlsx

For each subject folder, it finds *.csv files, infers run number from block=X, and writes:
sub-XXXX/ses-01/func/sub-XXXX_ses-01_task-ol_run-YY_events.tsv
into the BIDS root.

Usage:
    python batch_csv_to_bids_events.py --behav-root ~/PycharmProjects/ol_fmri/data/behavioral_data \
                                       --bids-root ~/PycharmProjects/ol_fmri/data/bids
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re
import json

def num(x):
    """Convert CSV entries like '--' to NaN, else float."""
    try:
        if isinstance(x, str) and x.strip() in {"--", ""}:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def convert_csv_to_events(csv_path, bids_root, subj, run, ses="01", task="ol", run_start=0.0):
    df = pd.read_csv(csv_path)

    def onset(col):
        v = df[col].map(num)
        return v - run_start

    events_rows = []

    # Button press events
    for t in onset("t_self_choice_on"):
        if pd.notna(t) and t >= 0:
            events_rows.append(dict(onset=t, duration=0.0, trial_type="self_choice"))

    # Other choice
    for t in onset("t_other_choice"):
        if pd.notna(t) and t >= 0:
            events_rows.append(dict(onset=t, duration=0.0, trial_type="other_choice"))

    # Other outcome (with reward as amplitude)
    if "partner_reward" in df.columns:
        reward = df["partner_reward"].astype(float)
    else:
        reward = pd.Series([np.nan]*len(df))

    for t, r in zip(onset("t_other_outcome"), reward):
        if pd.notna(t) and t >= 0:
            row = dict(onset=t, duration=0.0, trial_type="other_outcome")
            if pd.notna(r):
                row["amplitude"] = r
            events_rows.append(row)

    if not events_rows:
        print(f"⚠️ No events found in {csv_path}, skipping")
        return

    ev = pd.DataFrame(events_rows).sort_values("onset").reset_index(drop=True)
    if "amplitude" not in ev.columns:
        ev["amplitude"] = np.nan

    # Save to BIDS path
    func_dir = Path(bids_root) / f"sub-{subj}" / f"ses-{ses}" 
    func_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = func_dir / f"sub-{subj}_ses-{ses}_task-{task}_run-{run}_events.tsv"
    ev.to_csv(out_tsv, sep="\t", index=False, na_rep="n/a")
    print(f"✅ Wrote {out_tsv}")

    # Minimal sidecar
    out_json = func_dir / f"sub-{subj}_ses-{ses}_task-{task}_run-{run}_events.json"
    sidecar = {
        "onset": {"Description": "Event onset time in seconds from run start."},
        "duration": {"Description": "Event duration in seconds."},
        "trial_type": {"Description": "Condition label."},
        "amplitude": {"Description": "Parametric modulator (e.g., partner_reward)."}
    }
    out_json.write_text(json.dumps(sidecar, indent=2) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behav-root", required=True, help="Behavioral data root (with subject folders)")
    ap.add_argument("--bids-root", required=True, help="BIDS root where events.tsv will be written")
    args = ap.parse_args()

    behav_root = Path(args.behav_root)
    for subj_dir in behav_root.iterdir():
        if not subj_dir.is_dir():
            continue
        subj = subj_dir.name  # e.g., "0001" 
        # Skip >0010
        if not re.match(r"^\d{4}$", subj) or int(subj) > 10:
            print(f"Skipping {subj_dir}, not a valid subject folder")
            continue
        for csv_file in subj_dir.glob("*.csv"):
            # Infer run from "block=X.csv"
            m = re.search(r"block=(\d+)", str(csv_file))
            if not m:
                print(f"Skipping {csv_file}, no block number found")
                continue
            block = int(m.group(1))
            if block == 0:
                continue  # skip block=0 practice
            run = f"{block:02d}"  # block=0 → run-01
            convert_csv_to_events(csv_file, args.bids_root, subj, run)

if __name__ == "__main__":
    main()
