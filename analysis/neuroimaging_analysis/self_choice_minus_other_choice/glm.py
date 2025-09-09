import os, glob
import numpy as np
import pandas as pd
from pathlib import Path

from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import image, plotting
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.masking import apply_mask


class Setting:
    BASE_DIR = Path(__file__).resolve().parent # Directory of the current script
    NEURAL_DATA_DIR = BASE_DIR / "../../../data/derivatives/"
    EVENTS_DATA_DIR = BASE_DIR / "../../../data/events/"
    FIGURES_DIR = BASE_DIR / "figures"
    INTERIM_RESULTS_DIR = BASE_DIR / "interim_results"
    NEUROIMAGING_RESULTS_DIR = BASE_DIR / "neuroimaging_results"
    SUBJECTS = list(range(1, 5)) + list(range(6, 19)) + list(range(20, 23))  # Exclude subject 5 and 19 due to missing data
    INTERNAL_ID2SUBJECT_ID = {i+1: sub_id for i, sub_id in enumerate(SUBJECTS)}
    RUNS = [1, 2, 3, 4]

    @staticmethod
    def create_directories():
        Setting.INTERIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        Setting.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        Setting.NEUROIMAGING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_events(subject_id, run):
    events_path = Setting.EVENTS_DATA_DIR / f"sub-{subject_id:03d}" / f"sub-{subject_id:03d}_task-ol_run-{run:02d}_events.tsv"
    events = pd.read_table(events_path)
    events.query("trial_type == 'self_choice_on' or trial_type == 'other_choice'", inplace=True)
    events.drop(columns=["modulation"], axis=1, inplace=True)
    events.dropna(subset=["onset"], inplace=True)
    return events


def ensure_sample_mask(bold_file, sample_mask):
    import nibabel as nib
    if sample_mask is None:
        n_scans = nib.load(str(bold_file)).shape[-1]
        return np.arange(n_scans)  # keep all TRs
    return sample_mask


def load_bold_and_confounds(subject_id, run):
    func_path = Setting.NEURAL_DATA_DIR / f"sub-{subject_id:03d}" / "ses-01" / "func"
    bold_file = func_path / f"sub-{subject_id:03d}_ses-01_task-ol_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

    confound, sample_mask = load_confounds_strategy(
        str(bold_file),
        denoise_strategy="scrubbing",
        fd_threshold=0.5, std_dvars_threshold=1.5
    )
    sample_mask = ensure_sample_mask(str(bold_file), sample_mask)  # keep all if None
    return bold_file, confound, sample_mask


def save_design_matrix(first, subject_id):
    out_dir = Setting.INTERIM_RESULTS_DIR / f"sub-{subject_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_idx, X in enumerate(first.design_matrices_):
        X.to_csv(out_dir / f"sub-{subject_id:03d}_run-{run_idx+1:02d}_design_matrix.tsv",
                 sep="\t", index=False)
        

def save_contrast_maps(first, subject_id):
    out_dir = Setting.NEUROIMAGING_RESULTS_DIR / f"sub-{subject_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_regressors = first.design_matrices_[0].shape[1]
    contrasts = np.zeros(n_regressors)
    contrasts[0] = -1.0
    contrasts[1] = 1.0

    beta = first.compute_contrast(contrasts, output_type="effect_size")
    var  = first.compute_contrast(contrasts, output_type="effect_variance")
    zmap = first.compute_contrast(contrasts, output_type="z_score")

    beta_p = out_dir / f"self_choice_minus_other_choice_beta.nii.gz"
    var_p  = out_dir / f"self_choice_minus_other_choice_var.nii.gz"
    z_p    = out_dir / f"self_choice_minus_other_choice_zmap.nii.gz"

    beta.to_filename(beta_p)
    var.to_filename(var_p)
    zmap.to_filename(z_p)


def main():
    for subject_id in Setting.SUBJECTS:
        print(f"Processing Subject {subject_id:03d}...")

        events = []
        bold_files = []
        confounds = []
        sample_masks = []

        for run in Setting.RUNS:
            events.append(fetch_events(subject_id, run))

            bold_file, confound, sample_mask = load_bold_and_confounds(subject_id, run)
            bold_files.append(bold_file)
            confounds.append(confound)
            sample_masks.append(sample_mask)

        first = FirstLevelModel(
            t_r=0.8, hrf_model='spm', drift_model=None, # fMRIprep has coundered drift already
            high_pass=None, noise_model='ar1', smoothing_fwhm=6.0
        )

        print(f"Fitting first-level model for Subject {subject_id:03d}...")
        first = first.fit(bold_files, events=events, confounds=confounds, sample_masks=sample_masks)

        save_design_matrix(first, subject_id)
        save_contrast_maps(first, subject_id)
        print(f"Subject {subject_id:03d} processing complete.\n")

    print("All subjects processed.")


if __name__ == "__main__":
    Setting.create_directories()
    main()