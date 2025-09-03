#!/usr/bin/env python3
# plot_param_recovery.py
# Usage:
#   python plot_param_recovery.py --csv results.csv --outdir figures

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PARAM_SPECS = [
    ("learning_rate_for_reward", "Reward learning rate (α_r)"),
    ("learning_rate_for_action", "Action learning rate (α_a)"),
    ("relative_weight_of_action", "Relative weight of action (w_A)"),
    ("beta", "Inverse temperature (β)"),
]

def _limits_with_padding(true_vals, est_vals, pad_frac=0.05):
    lo = float(min(np.min(true_vals), np.min(est_vals)))
    hi = float(max(np.max(true_vals), np.max(est_vals)))
    span = hi - lo
    pad = pad_frac * (span if span > 0 else 1.0)
    return lo - pad, hi + pad

def _pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def plot_one(ax, t, e, label):
    lo, hi = _limits_with_padding(t, e)
    ax.scatter(t, e)
    ax.plot([lo, hi], [lo, hi])  # y = x
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Estimated {label}")
    ax.set_title(f"True vs Estimated — {label}")
    ax.set_aspect('equal', adjustable='box')

def main():
    # p = argparse.ArgumentParser()
    # p.add_argument("--csv", required=True, help="Path to CSV with columns true_* and estimated_*")
    # p.add_argument("--outdir", default="figures", help="Directory to save figures/diagnostics")
    # args = p.parse_args()

    outdir = Path("results/computational_analysis") 
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("results/computational_analysis/full_model_parameter_recovery_results.csv")

    diagnostics = []
    pdf_path = outdir / "parameter_recovery_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for key, label in PARAM_SPECS:
            tcol = f"true_{key}"
            ecol = f"estimated_{key}"

            t = df[tcol].to_numpy()
            e = df[ecol].to_numpy()

            if key == "beta":
                # For beta, we take the log to better visualize the range
                t = np.log(t + 1e-6)  # Avoid log(0)
                e = np.log(e + 1e-6)  # Avoid log(0)
                label = "Log " + label

            # Plot
            fig, ax = plt.subplots()
            plot_one(ax, t, e, label)
            png_path = outdir / f"scatter_{key}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Diagnostics
            r = _pearsonr(t, e)
            rmse = float(np.sqrt(np.mean((e - t) ** 2)))
            diagnostics.append({
                "parameter": key,
                "label": label,
                "pearson_r": r,
                "rmse": rmse,
                "n": len(t),
                "png_path": str(png_path)
            })

    # Save diagnostics table
    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(outdir / "diagnostics.csv", index=False)

    print(f"Saved individual PNGs to: {outdir}")
    print(f"Saved multi-page PDF to: {pdf_path}")
    print(f"Saved diagnostics CSV to: {outdir / 'diagnostics.csv'}")

if __name__ == "__main__":
    main()
