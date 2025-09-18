import json
import numpy as np
import pandas as pd
from pathlib import Path

from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.glm.first_level import FirstLevelModel


class Setting:
    BASE_DIR = Path(__file__).resolve().parent # Directory of the current script
    NEURAL_DATA_DIR = BASE_DIR / "../../../data/derivatives/"
    EVENTS_DATA_PATH = BASE_DIR / "../../../data/events/"
    CONDITION_DATA_PATH = (BASE_DIR / "../../../data/preprocessed/conditions.csv").resolve()

    INTERIM_RESULTS_DIR = BASE_DIR / "interim_results"
    FIGURES_DIR = BASE_DIR / "figures"
    NEUROIMAGING_RESULTS_DIR = BASE_DIR / "neuroimaging_results"
    SUBJECTS = list(range(1, 5)) + list(range(6, 19)) + list(range(20, 26)) + list(range(27, 38))  # Exclude subject 5, 19, and 26
    INTERNAL_ID2SUBJECT_ID = {i+1: sub_id for i, sub_id in enumerate(SUBJECTS)}
    RUNS = [1, 2, 3, 4]

    CONFOUND_COLS = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]

    @staticmethod
    def create_directories():
        Setting.INTERIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        Setting.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        Setting.NEUROIMAGING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        for sub in Setting.SUBJECTS:
            (Setting.INTERIM_RESULTS_DIR / f"sub-{sub:03d}").mkdir(parents=True, exist_ok=True)


def fetch_events(subject_id, run):
    events_path = Setting.EVENTS_DATA_PATH / f"sub-{subject_id:03d}" / f"sub-{subject_id:03d}_task-ol_run-{run:02d}_events.tsv"
    events = pd.read_csv(events_path, sep="\t")
    events.query("trial_type != 'rpe_q0' and trial_type != 'other_choice' and trial_type != 'other_outcome'", inplace=True)
    events.dropna(inplace=True)

    return events


def ensure_sample_mask(bold_file, sample_mask):
    import nibabel as nib
    if sample_mask is None:
        n_scans = nib.load(str(bold_file)).shape[-1]
        return np.arange(1, n_scans)  # keep all TRs
    
    return sample_mask


def load_bold_and_confounds(subject_id, run, used_columns=None):
    func_path = Setting.NEURAL_DATA_DIR / f"sub-{subject_id:03d}" / "ses-01" / "func"
    bold_file = func_path / f"sub-{subject_id:03d}_ses-01_task-ol_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

    try:
        confound, sample_mask = load_confounds_strategy(
            str(bold_file),
        )
    except json.decoder.JSONDecodeError:
        return None, None, None
    
    if used_columns is not None:
        confound = confound[used_columns]

    sample_mask = ensure_sample_mask(str(bold_file), sample_mask)  # keep all if None

    return bold_file, confound, sample_mask


def save_design_matrix(first, subject_id, noise_condition):
    out_dir = Setting.INTERIM_RESULTS_DIR / f"sub-{subject_id:03d}" / f"condition-{noise_condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_idx, X in enumerate(first.design_matrices_):
        X.to_csv(out_dir / f"sub-{subject_id:03d}_run-{run_idx+1:02d}_design_matrix.tsv",
                 sep="\t", index=False)


def save_contrast_maps(first, subject_id, noise_condition):
    out_dir = Setting.NEUROIMAGING_RESULTS_DIR / f"sub-{subject_id:03d}" / f"condition-{noise_condition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = {
        "RPE": "rpe",
        "APE": "ape",
    }

    rows = []
    for name, reg in con.items():
        beta = first.compute_contrast(reg, output_type="effect_size")      # β map
        var  = first.compute_contrast(reg, output_type="effect_variance")  # var(β)
        zmap = first.compute_contrast(reg, output_type="z_score")          # Z

        beta_p = out_dir / f"{name}_beta.nii.gz"
        var_p  = out_dir / f"{name}_var.nii.gz"
        z_p    = out_dir / f"{name}_zmap.nii.gz"

        beta.to_filename(beta_p); var.to_filename(var_p); zmap.to_filename(z_p)
        rows.append({
            "subject_label": f"sub-{subject_id:03d}",
            "map_name": name,
            "effect_map": str(beta_p),
            "variance_map": str(var_p),
            "z_map": str(z_p),
        })

    pd.DataFrame(rows).to_csv(out_dir / "manifest.tsv", sep="\t", index=False)


def main(is_skipped=False):
    noise_level_condition = pd.read_csv(Setting.CONDITION_DATA_PATH)
    for subject_id in Setting.SUBJECTS:
        print(f"Processing Subject {subject_id:03d}...")

        neural_result = Setting.NEUROIMAGING_RESULTS_DIR / f"sub-{subject_id:03d}" / "condition-1" / "RPE_beta.nii.gz"
        if is_skipped and neural_result.exists():
            print(f"Subject {subject_id:03d} already processed, skipping.")
            continue

        events = {1: [], 2: []}
        bold_files = {1: [], 2: []}
        confounds = {1: [], 2: []}
        sample_masks = {1: [], 2: []}

        for run in Setting.RUNS:
            noise_condition = noise_level_condition.query(f"subject_id == {subject_id} and run == {run}")["condition"].values[0]

            bold_file, confound, sample_mask = load_bold_and_confounds(subject_id, run, Setting.CONFOUND_COLS)
            if bold_file is None:
                print(f"  Warning: Could not load BOLD or confounds for Subject {subject_id:03d}, Run {run:02d}. Skipping this run.")
                continue
            bold_files[noise_condition].append(bold_file)
            confounds[noise_condition].append(confound)
            sample_masks[noise_condition].append(sample_mask)
            events[noise_condition].append(fetch_events(subject_id, run))

        first = FirstLevelModel(
            t_r=0.8, hrf_model='spm', drift_model=None,
            high_pass=None, noise_model='ar1', smoothing_fwhm=8.0
        )

        print(f"Fitting first-level model for Subject {subject_id:03d}...")
    
        for noise_condition in [1, 2]:
            first = first.fit(bold_files[noise_condition], events=events[noise_condition], confounds=confounds[noise_condition])

            save_design_matrix(first, subject_id, noise_condition)
            save_contrast_maps(first, subject_id, noise_condition)
        
        print(f"Subject {subject_id:03d} processing complete.\n")

    print("All subjects processed.")


if __name__ == "__main__":
    Setting.create_directories()
    main(False)
