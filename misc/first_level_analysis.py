from pathlib import Path
import pandas as pd
import numpy as np
import sys

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn import plotting

DERIV_ROOT = Path("/Users/morishitag/PycharmProjects/ol_fmri/data/derivatives")
BIDS_ROOT  = Path("/Users/morishitag/PycharmProjects/ol_fmri/data/bids")

SESSION = "01"
TASK = "ol"
RUNS = ["01","02","03","04"]

COND_SELF, COND_OTHER = "self_choice", "other_choice"
SMOOTH_FWHM = 6.0
P_VOX = 0.001
K_EXTENT = 5

OUT_DIR = Path("./glm_self_other_out_fixed")
OUT_DIR.mkdir(exist_ok=True)

CONFOUND_COLS = [
    "trans_x","trans_y","trans_z","rot_x","rot_y","rot_z",
    "csf","white_matter","global_signal","framewise_displacement"
]

def sub3_to_sub4(s3: str) -> str:
    return f"{int(s3):04d}"   # 001 -> 0001

def list_subjects():
    # ★ ディレクトリのみを対象にする（*.html を除外）
    subs = []
    for p in sorted(DERIV_ROOT.glob("sub-*")):
        if p.is_dir() and p.name.startswith("sub-"):
            subs.append(p.name.split("-")[1])
    return subs

import re

def get_paths(sub3: str, run: str):
    func_dir = DERIV_ROOT / f"sub-{sub3}" / f"ses-{SESSION}" / "func"

    # BOLD
    cand = list(func_dir.glob(
        f"sub-{sub3}_ses-{SESSION}_task-{TASK}_run-{run}_*desc-preproc_bold.nii.gz"
    ))
    if not cand:
        cand = list(func_dir.glob(
            f"sub-{sub3}_ses-{SESSION}_task-{TASK}_run-{run}_desc-preproc_bold.nii.gz"
        ))
    if len(cand) != 1:
        listing = "\n  - ".join(sorted(p.name for p in func_dir.glob("*")))
        raise FileNotFoundError(
            f"[BOLD not found or not unique]\n  sub-{sub3} run-{run}\n  looked in: {func_dir}\n  found:\n  - {listing}"
        )
    bold = cand[0]

    # --- confounds ---
    # bold ファイル名から space-XXX を削除
    base_name = re.sub(r"_space-[^_]+", "", bold.name)
    # preproc_bold を confounds に置換
    conf_name = base_name.replace("_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv")
    conf_name = conf_name.replace("_desc-preproc_bold.nii", "_desc-confounds_timeseries.tsv")
    conf = func_dir / conf_name
    if not conf.exists():
        raise FileNotFoundError(f"[Confounds not found]\n expected: {conf}")

    # events
    sub4 = f"{int(sub3):04d}"
    ev = BIDS_ROOT / f"sub-{sub4}" / f"ses-{SESSION}" / f"sub-{sub4}_ses-{SESSION}_task-{TASK}_run-{run}_events.tsv"
    if not ev.exists():
        raise FileNotFoundError(f"[Events missing] {ev}")

    return bold, conf, ev


def load_confounds(path: Path):
    df = pd.read_csv(path, sep="\t")
    keep = [c for c in CONFOUND_COLS if c in df.columns]
    keep += [c for c in df.columns if "outlier" in c]
    return df[keep].fillna(0) if keep else None

# -------- 1st level --------
first_level_maps = []
SUBS = list_subjects()

for sub3 in SUBS:
    imgs, evs, confs = [], [], []
    for run in RUNS:
        bold, conf, ev = get_paths(sub3, run)
        ev_df = pd.read_csv(ev, sep="\t")
        ev_df = ev_df[ev_df["trial_type"].isin([COND_SELF, COND_OTHER])]
        if ev_df.empty:
            continue
        if "duration" not in ev_df: ev_df["duration"] = 1.0
        ev_df["duration"] = ev_df["duration"].fillna(1.0).replace(0,1.0)
        imgs.append(str(bold))
        evs.append(ev_df)
        confs.append(load_confounds(conf))
    if not imgs:
        print(f"[WARN] usable runs not found for sub-{sub3}; skipping.", file=sys.stderr)
        continue

    glm = FirstLevelModel(
        t_r=0.8, slice_time_ref=0.5,
        hrf_model="spm", drift_model="cosine", high_pass=0.008,
        smoothing_fwhm=SMOOTH_FWHM, noise_model="ar1",
        standardize=False, minimize_memory=True
    ).fit(imgs, events=evs, confounds=confs)

    tmap = glm.compute_contrast(f"{COND_SELF} - {COND_OTHER}", output_type="stat")
    out_t = OUT_DIR / f"sub-{sub3}_self-minus-other_tmap.nii.gz"
    tmap.to_filename(out_t)
    first_level_maps.append(str(out_t))
    print(f"[OK] 1st-level done: sub-{sub3}")

